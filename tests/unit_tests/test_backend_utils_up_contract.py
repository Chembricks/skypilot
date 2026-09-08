"""The UP-status contract for _update_cluster_status.

check_cluster_available() validates a cluster in two independent steps: the
recorded status must be UP, and *then*

    if handle.head_ip is None:
        raise exceptions.ClusterNotUpError(
            f'Cluster {cluster_name!r} has been stopped or not properly '
            'set up. ...')

So the second step assumes an invariant the first one never states: a cluster
recorded UP carries a handle with usable connection info. When that breaks,
`sky status` reports UP while `sky queue` raises ClusterNotUpError on the same
cluster -- not a cache skew, but the invariant being violated where UP is
written.

These tests pin the invariant itself rather than any single mechanism that can
violate it, so a short-circuit added to the promotion condition later trips
them too.
"""
import time
from unittest import mock

import pytest

from sky import backends
from sky import clouds
from sky.backends import backend_utils
from sky.utils import status_lib


def _make_handle(*, has_ray, has_ips, cloud):
    handle = mock.Mock(spec=backends.CloudVmRayResourceHandle)
    handle.cluster_name = 'test-cluster'
    handle.cluster_name_on_cloud = 'test-cluster-1234'
    handle.cluster_yaml = '/fake/path/cluster.yaml'
    handle.launched_nodes = 1
    handle.num_ips_per_node = 1
    handle.cached_cluster_info = None
    handle.launched_resources = mock.Mock(unsafe=True)
    handle.launched_resources.cloud = cloud
    handle.launched_resources.use_spot = False
    handle.launched_resources.assert_launchable.return_value = (
        handle.launched_resources)
    handle.provision_runtime_metadata = mock.Mock()
    handle.provision_runtime_metadata.has_ray = has_ray

    # A handle whose launch never reached update_cluster_ips() has no IPs, so
    # head_ip is None and get_command_runners() yields nothing -- exactly the
    # state the backend persists while the cluster is still INIT.
    if has_ips:
        handle.stable_internal_external_ips = [('10.0.0.1', '1.2.3.4')]
        handle.cached_external_ips = ['1.2.3.4']
        handle.head_ip = '1.2.3.4'
        runner = mock.Mock()
        # get_node_counts_from_ray_status() is nested in _update_cluster_status,
        # so a healthy `ray status` is faked at the runner instead.
        runner.run.return_value = (0, 'ray-status-output', '')
        handle.get_command_runners.return_value = [runner]
    else:
        handle.stable_internal_external_ips = None
        handle.cached_external_ips = None
        handle.head_ip = None
        handle.get_command_runners.return_value = []
    return handle


def _refresh(handle, node_statuses):
    """Run _update_cluster_status; return the `ready` flag it persisted."""
    record = {
        'handle': handle,
        'status': status_lib.ClusterStatus.INIT,
        'cluster_hash': 'fake-hash',
        'autostop': -1,
        'to_down': False,
        'launched_at': time.time() - 3600,
    }
    persisted = {}

    def _capture(cluster_name, cluster_handle, requested_resources, ready,
                 **kwargs):
        persisted['ready'] = ready

    external_failure = mock.Mock()
    external_failure.get.return_value = None
    backend = mock.Mock(spec=backends.CloudVmRayBackend)
    backend.is_definitely_autostopping.return_value = False

    with mock.patch.object(backend_utils,
                           '_query_cluster_status_via_cloud_api',
                           return_value=node_statuses), \
         mock.patch.object(backend_utils, 'ExternalFailureSource',
                           external_failure), \
         mock.patch.object(backend_utils, 'get_backend_from_handle',
                           return_value=backend), \
         mock.patch.object(backend_utils, '_count_healthy_nodes_from_ray',
                           return_value=(1, 0)), \
         mock.patch.object(backend_utils.global_user_state,
                           'add_cluster_event'), \
         mock.patch.object(backend_utils.global_user_state,
                           'add_or_update_cluster',
                           side_effect=_capture), \
         mock.patch.object(backend_utils.global_user_state,
                           'get_cluster_from_name',
                           return_value=record):
        backend_utils._update_cluster_status('test-cluster',
                                             record,
                                             retry_if_missing=False)
    return persisted.get('ready')


ALL_NODES_UP = {'node-0': (status_lib.ClusterStatus.UP, None)}


class TestUpStatusContract:
    """A cluster is promoted to UP only if its handle can actually be used."""

    @pytest.mark.parametrize('uses_ray', [True, False])
    @pytest.mark.parametrize('has_ray', [True, False])
    @pytest.mark.parametrize('has_ips', [True, False])
    def test_up_implies_usable_handle(self, uses_ray, has_ray, has_ips):
        # The cloud API reporting every node up must never be sufficient on its
        # own: whatever combination of runtime metadata gates the health check,
        # a cluster recorded UP has to satisfy what check_cluster_available()
        # demands of it.
        with mock.patch.object(clouds.Vast, 'uses_ray',
                               classmethod(lambda cls: uses_ray)):
            handle = _make_handle(has_ray=has_ray,
                                  has_ips=has_ips,
                                  cloud=clouds.Vast())
            ready = _refresh(handle, ALL_NODES_UP)
        if ready:
            assert handle.head_ip is not None, (
                f'promoted to UP with head_ip=None '
                f'(uses_ray={uses_ray}, has_ray={has_ray}); '
                'check_cluster_available() will reject this cluster, so '
                '`sky status` and `sky queue` will disagree')

    def test_interrupted_launch_stays_init(self):
        # Narrow regression for the has_ray=False short-circuit: a handle the
        # backend persisted before provisioning finished, on a host whose VM is
        # genuinely running, must not be promoted on the cloud API alone.
        handle = _make_handle(has_ray=False, has_ips=False, cloud=clouds.Vast())
        assert _refresh(handle, ALL_NODES_UP) is not True

    def test_healthy_cluster_still_reaches_up(self):
        # Guard against fixing the above by never promoting anything.
        handle = _make_handle(has_ray=True, has_ips=True, cloud=clouds.Vast())
        assert _refresh(handle, ALL_NODES_UP) is True
