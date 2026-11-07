from functools import cached_property

from src.services.system_health_service import HealthCheckService
from src.services.system_info_service import InfoService


class Structure:
    @cached_property
    def system_info_service(self) -> InfoService:
        return InfoService()

    @cached_property
    def health_check(self) -> HealthCheckService:
        return HealthCheckService()


structure = Structure()
