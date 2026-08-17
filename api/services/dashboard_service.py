"""
Dashboard Service
"""

import logging
from .health_service import HealthService
from .nps_service import NPSService

logger = logging.getLogger(__name__)

class DashboardService:
    def __init__(self):
        self.health_service = HealthService()
        self.nps_service = NPSService()
    
    def get_dashboard(self, input_data: dict):
        """Get full dashboard data"""
        try:
            # Get health prediction
            health_result = self.health_service.predict(input_data)
            
            # Prepare NPS input
            nps_input = {
                'operational_health': health_result['operational_health'],
                'business_intelligence_factor': input_data.get('business_intelligence_factor', 0.5),
                'member_intelligence_factor': input_data.get('member_intelligence_factor', 0.5),
                'target_release_rate': input_data.get('target_release_rate', 75),
                'actual_release_rate': input_data.get('actual_release_rate', 75),
                'release_gap': input_data.get('target_release_rate', 75) - input_data.get('actual_release_rate', 75),
                'release_delta': 0,  # Would need history
                'total_calls_received': input_data.get('total_calls_received', 2000)
            }
            
            # Get NPS prediction
            nps_result = self.nps_service.predict(nps_input)
            
            # Combine
            return {
                "system_status": {
                    "health_engine": "active",
                    "nps_engine": "active",
                    "status": "ok"
                },
                "target_profile": {
                    "quality": input_data.get('target_quality'),
                    "competency": input_data.get('target_competency'),
                    "attendance": input_data.get('target_attendance'),
                    "release_rate": input_data.get('target_release_rate'),
                    "transfer_rate": input_data.get('target_transfer_rate')
                },
                "today_performance": {
                    "quality": input_data.get('actual_quality'),
                    "competency": input_data.get('actual_competency'),
                    "attendance": input_data.get('actual_attendance'),
                    "release_rate": input_data.get('actual_release_rate'),
                    "transfer_rate": input_data.get('actual_transfer_rate'),
                    "total_calls": input_data.get('total_calls_received')
                },
                "intelligence_signals": {
                    "oif": input_data.get('operational_intelligence_factor'),
                    "bif": input_data.get('business_intelligence_factor'),
                    "mif": input_data.get('member_intelligence_factor')
                },
                "tomorrow_forecast": {
                    "operational_health": health_result['operational_health'],
                    "nps": nps_result['nps'],
                    "promoters": nps_result['promoters'],
                    "passives": nps_result['passives'],
                    "detractors": nps_result['detractors'],
                    "score_counts": nps_result['score_counts']
                }
            }
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
            raise

# Module-level compatibility surface
def get_dashboard(input_data: dict):
    return DashboardService().get_dashboard(input_data)
