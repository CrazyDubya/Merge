"""Prometheus metrics for monitoring Village framework."""

from typing import Optional

try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Info,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class VillageMetrics:
    """Prometheus metrics for Village framework.

    Tracks various metrics including:
    - LLM API calls and latency
    - Villager task processing
    - Village collaboration
    - Rate limiting and quota usage
    - Storage operations
    """

    def __init__(self, registry: Optional[any] = None) -> None:
        """Initialize metrics.

        Args:
            registry: Optional Prometheus registry (creates new if not provided)

        Raises:
            ImportError: If prometheus_client is not installed
        """
        if not PROMETHEUS_AVAILABLE:
            raise ImportError(
                "prometheus_client not installed. "
                "Install with: pip install prometheus-client>=0.17.0"
            )

        self.registry = registry or CollectorRegistry()

        # Info metric for version tracking
        self.village_info = Info(
            'village_framework',
            'Village framework information',
            registry=self.registry
        )
        self.village_info.info({
            'version': '0.1.0',
            'phase': '1'
        })

        # LLM Provider Metrics
        self.llm_requests_total = Counter(
            'village_llm_requests_total',
            'Total LLM API requests',
            ['provider', 'model', 'method', 'status'],
            registry=self.registry
        )

        self.llm_request_duration = Histogram(
            'village_llm_request_duration_seconds',
            'LLM API request duration in seconds',
            ['provider', 'model', 'method'],
            registry=self.registry,
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
        )

        self.llm_tokens_used = Counter(
            'village_llm_tokens_total',
            'Total LLM tokens used',
            ['provider', 'model', 'type'],  # type: prompt, completion, total
            registry=self.registry
        )

        self.llm_errors_total = Counter(
            'village_llm_errors_total',
            'Total LLM errors',
            ['provider', 'model', 'error_type'],
            registry=self.registry
        )

        # Villager Metrics
        self.villagers_active = Gauge(
            'village_villagers_active',
            'Number of active villagers',
            ['village'],
            registry=self.registry
        )

        self.villager_tasks_total = Counter(
            'village_villager_tasks_total',
            'Total tasks processed by villagers',
            ['villager', 'village', 'status'],
            registry=self.registry
        )

        self.villager_task_duration = Histogram(
            'village_villager_task_duration_seconds',
            'Villager task processing duration',
            ['villager', 'village'],
            registry=self.registry,
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
        )

        # Village Metrics
        self.villages_total = Gauge(
            'village_villages_total',
            'Total number of villages',
            registry=self.registry
        )

        self.collaboration_requests_total = Counter(
            'village_collaboration_requests_total',
            'Total collaboration requests',
            ['village', 'status'],
            registry=self.registry
        )

        self.collaboration_duration = Histogram(
            'village_collaboration_duration_seconds',
            'Village collaboration duration',
            ['village'],
            registry=self.registry,
            buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
        )

        # Rate Limiting Metrics
        self.rate_limit_hits_total = Counter(
            'village_rate_limit_hits_total',
            'Total rate limit hits',
            ['resource'],
            registry=self.registry
        )

        self.rate_limit_tokens_available = Gauge(
            'village_rate_limit_tokens_available',
            'Available rate limit tokens',
            ['resource'],
            registry=self.registry
        )

        # Quota Metrics
        self.quota_usage = Gauge(
            'village_quota_usage',
            'Current quota usage',
            ['period'],  # hourly, daily, monthly
            registry=self.registry
        )

        self.quota_limit = Gauge(
            'village_quota_limit',
            'Quota limit',
            ['period'],
            registry=self.registry
        )

        # Storage Metrics
        self.storage_operations_total = Counter(
            'village_storage_operations_total',
            'Total storage operations',
            ['operation', 'type', 'status'],  # type: village, villager, memory, task
            registry=self.registry
        )

        self.storage_operation_duration = Histogram(
            'village_storage_operation_duration_seconds',
            'Storage operation duration',
            ['operation', 'type'],
            registry=self.registry,
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0)
        )

        self.storage_size_bytes = Gauge(
            'village_storage_size_bytes',
            'Storage size in bytes',
            ['type'],
            registry=self.registry
        )

        # Memory Metrics
        self.memory_entries_total = Gauge(
            'village_memory_entries_total',
            'Total memory entries',
            ['owner_type'],  # village, villager
            registry=self.registry
        )

        # Communication Metrics
        self.messages_sent_total = Counter(
            'village_messages_sent_total',
            'Total inter-villager messages sent',
            ['from_villager', 'to_villager'],
            registry=self.registry
        )

        # Error Metrics
        self.errors_total = Counter(
            'village_errors_total',
            'Total errors',
            ['component', 'error_type'],
            registry=self.registry
        )

    def export_metrics(self) -> bytes:
        """Export metrics in Prometheus format.

        Returns:
            Metrics in Prometheus text format
        """
        return generate_latest(self.registry)

    def get_content_type(self) -> str:
        """Get Prometheus content type.

        Returns:
            Content type string
        """
        return CONTENT_TYPE_LATEST


# Global metrics instance (lazy initialization)
_metrics_instance: Optional[VillageMetrics] = None


def get_metrics() -> Optional[VillageMetrics]:
    """Get global metrics instance.

    Returns:
        VillageMetrics instance or None if not initialized
    """
    return _metrics_instance


def initialize_metrics(registry: Optional[any] = None) -> VillageMetrics:
    """Initialize global metrics instance.

    Args:
        registry: Optional Prometheus registry

    Returns:
        Initialized VillageMetrics instance

    Raises:
        ImportError: If prometheus_client is not installed
    """
    global _metrics_instance

    if _metrics_instance is None:
        _metrics_instance = VillageMetrics(registry)

    return _metrics_instance


def metrics_available() -> bool:
    """Check if Prometheus metrics are available.

    Returns:
        True if prometheus_client is installed
    """
    return PROMETHEUS_AVAILABLE
