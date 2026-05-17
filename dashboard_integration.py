"""
Integration layer between Streamlit dashboard and talent mobility engine.
This module provides clean interfaces for the dashboard to interact with your existing engine.
"""

from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime

# Import your engine (adjust path as needed)
try:
    from src.talent_agent.engine import process_batch
    from src.talent_agent.io import save_uploaded_file
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False
    print("⚠️  Engine not available - using demo mode")


class DashboardIntegration:
    """Main integration class for connecting dashboard to engine"""
    
    def __init__(self, temp_dir: str = "data/dashboard_temp", output_dir: str = "outputs/dashboard"):
        self.temp_dir = Path(temp_dir)
        self.output_dir = Path(output_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_hr_batch(
        self,
        resume_files: List,
        jd_files: List,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process HR batch analysis with multiple resumes and JDs.
        
        Args:
            resume_files: List of uploaded resume files (Streamlit UploadedFile objects)
            jd_files: List of uploaded JD files
            config: Configuration dict with keys:
                - enable_redaction: bool
                - enable_bias_scan: bool
                - model: str
                - score_mode: str
                - semantic_weight: float
                - min_confidence: float
        
        Returns:
            Formatted results dict ready for dashboard display
        """
        if not ENGINE_AVAILABLE:
            return self._get_demo_hr_results()
        
        # Save uploaded files
        employees = []
        for i, resume_file in enumerate(resume_files, start=1):
            resume_path = self._save_upload(resume_file, f"resume_{i}")
            employees.append({
                "employee_id": f"E{i:03d}",
                "name": self._extract_name_from_filename(resume_file.name),
                "resume_path": str(resume_path)
            })
        
        roles = []
        for i, jd_file in enumerate(jd_files, start=1):
            jd_path = self._save_upload(jd_file, f"jd_{i}")
            roles.append({
                "role_id": f"R{i:03d}",
                "role_title": self._extract_name_from_filename(jd_file.name),
                "jd_path": str(jd_path)
            })
        
        # Run engine
        run_id = f"hr_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = self.output_dir / run_id
        
        results = process_batch(
            employees=employees,
            roles=roles,
            run_dir=str(run_dir),
            enable_redaction=config.get("enable_redaction", True),
            enable_bias_scan=config.get("enable_bias_scan", True),
            model=config.get("model", "gpt-4o-mini"),
            score_mode=config.get("score_mode", "hybrid"),
            semantic_weight=config.get("semantic_weight", 0.40),
            min_confidence=config.get("min_confidence", 0.55)
        )
        
        # Format for dashboard
        return self._format_hr_results(results, roles)
    
    def process_employee_analysis(
        self,
        employee_id: str,
        resume_file=None,
        current_jd_file=None,
        target_jd_files: List = None,
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process single employee analysis for self-service portal.
        
        Args:
            employee_id: Employee identifier
            resume_file: Optional updated resume (Streamlit UploadedFile)
            current_jd_file: Optional current role JD
            target_jd_files: List of target role JDs to match against
            config: Configuration dict
        
        Returns:
            Formatted results for employee view
        """
        if not ENGINE_AVAILABLE:
            return self._get_demo_employee_results(employee_id)
        
        config = config or {}
        
        # Prepare employee data
        employee = {
            "employee_id": employee_id,
            "name": self._lookup_employee_name(employee_id)
        }
        
        if resume_file:
            resume_path = self._save_upload(resume_file, f"emp_{employee_id}_resume")
            employee["resume_path"] = str(resume_path)
        else:
            # Try to find existing resume
            employee["resume_path"] = self._find_existing_resume(employee_id)
        
        if current_jd_file:
            current_jd_path = self._save_upload(current_jd_file, f"emp_{employee_id}_current_jd")
            employee["current_jd_path"] = str(current_jd_path)
        else:
            employee["current_jd_path"] = self._find_existing_current_jd(employee_id)
        
        # Prepare target roles
        roles = []
        if target_jd_files:
            for i, jd_file in enumerate(target_jd_files, start=1):
                jd_path = self._save_upload(jd_file, f"target_jd_{i}")
                roles.append({
                    "role_id": f"T{i:03d}",
                    "role_title": self._extract_name_from_filename(jd_file.name),
                    "jd_path": str(jd_path)
                })
        else:
            # Use default set of open roles
            roles = self._get_default_open_roles()
        
        # Run engine
        run_id = f"emp_{employee_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = self.output_dir / run_id
        
        results = process_batch(
            employees=[employee],
            roles=roles,
            run_dir=str(run_dir),
            enable_redaction=config.get("enable_redaction", True),
            enable_bias_scan=config.get("enable_bias_scan", True),
            model=config.get("model", "gpt-4o-mini"),
            deep_dive_top_n=3,  # Get detailed info for top 3
            score_mode=config.get("score_mode", "hybrid"),
            semantic_weight=config.get("semantic_weight", 0.40),
            min_confidence=config.get("min_confidence", 0.55)
        )
        
        # Format for employee view
        return self._format_employee_results(results, employee_id)
    
    def load_existing_results(self, run_id: str) -> Dict[str, Any]:
        """Load previously computed results"""
        result_file = self.output_dir / run_id / f"batch_results_{run_id}.json"
        
        if not result_file.exists():
            raise FileNotFoundError(f"Results not found: {result_file}")
        
        with open(result_file, 'r') as f:
            return json.load(f)
    
    # ========================================================================
    # Private helper methods
    # ========================================================================
    
    def _save_upload(self, uploaded_file, prefix: str) -> Path:
        """Save an uploaded file to temp directory"""
        filename = f"{prefix}_{uploaded_file.name}"
        path = self.temp_dir / filename
        
        with open(path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        return path
    
    def _extract_name_from_filename(self, filename: str) -> str:
        """Extract a clean name from filename"""
        name = Path(filename).stem
        # Remove common prefixes/suffixes
        name = name.replace('_resume', '').replace('Resume', '')
        name = name.replace('_JD', '').replace('JD', '')
        name = name.replace('_', ' ').replace('-', ' ')
        return name.strip()
    
    def _lookup_employee_name(self, employee_id: str) -> str:
        """Lookup employee name from ID (stub - replace with actual lookup)"""
        # TODO: Replace with actual employee database lookup
        demo_names = {
            "E001": "Sarah Johnson",
            "E002": "Michael Chen",
            "E003": "Emily Rodriguez"
        }
        return demo_names.get(employee_id, f"Employee {employee_id}")
    
    def _find_existing_resume(self, employee_id: str) -> str:
        """Find existing resume for employee (stub)"""
        # TODO: Replace with actual file lookup
        possible_path = Path(f"data/resumes/{employee_id}_resume.docx")
        return str(possible_path) if possible_path.exists() else ""
    
    def _find_existing_current_jd(self, employee_id: str) -> str:
        """Find existing current JD (stub)"""
        # TODO: Replace with actual file lookup
        possible_path = Path(f"data/current_jds/{employee_id}_current.docx")
        return str(possible_path) if possible_path.exists() else ""
    
    def _get_default_open_roles(self) -> List[Dict]:
        """Get default list of open roles (stub)"""
        # TODO: Replace with actual open roles from database/file
        return [
            {"role_id": "R001", "role_title": "Senior Data Scientist", 
             "jd_path": "data/jds/senior_data_scientist.docx"},
            {"role_id": "R002", "role_title": "Product Manager",
             "jd_path": "data/jds/product_manager.docx"},
            {"role_id": "R003", "role_title": "Business Intelligence Lead",
             "jd_path": "data/jds/bi_lead.docx"}
        ]
    
    def _format_hr_results(self, results: Dict, roles: List[Dict]) -> Dict:
        """Format engine results for HR dashboard display"""
        formatted = {
            "summary": {
                "total_candidates": len(results["matches"]),
                "total_roles": len(roles),
                "avg_match_score": self._calculate_avg_score(results["matches"]),
                "ready_to_move": self._count_ready_candidates(results["matches"])
            },
            "roles": {},
            "candidates": [],
            "errors": results.get("errors", [])
        }
        
        # Group candidates by role
        for role in roles:
            role_id = role["role_id"]
            formatted["roles"][role_id] = {
                "role_id": role_id,
                "title": role["role_title"],
                "top_candidates": []
            }
        
        # Add top 3 candidates for each role
        for match in results["matches"]:
            candidate_info = {
                "employee_id": match["employee_id"],
                "name": match["employee_name"],
                "ranked_roles": match.get("ranked_roles", [])
            }
            formatted["candidates"].append(candidate_info)
            
            # Add to each role's top candidates
            for ranked_role in match.get("ranked_roles", []):
                role_id = ranked_role["role_id"]
                if role_id in formatted["roles"]:
                    formatted["roles"][role_id]["top_candidates"].append({
                        "employee_id": match["employee_id"],
                        "name": match["employee_name"],
                        "match_score": ranked_role.get("score", {}),
                        "missing_skills": ranked_role.get("missing_must_haves", []),
                        "bias_severity": ranked_role.get("bias_report", {}).get("severity", "none")
                    })
        
        # Sort top candidates for each role
        for role_id in formatted["roles"]:
            candidates = formatted["roles"][role_id]["top_candidates"]
            candidates.sort(
                key=lambda x: x["match_score"].get("overall", 0),
                reverse=True
            )
            formatted["roles"][role_id]["top_candidates"] = candidates[:3]
        
        return formatted
    
    def _format_employee_results(self, results: Dict, employee_id: str) -> Dict:
        """Format engine results for employee view"""
        if not results["matches"]:
            return {"error": "No matches found"}
        
        match = results["matches"][0]  # Single employee
        
        formatted = {
            "employee_id": employee_id,
            "name": match["employee_name"],
            "top_roles": [],
            "career_metrics": {
                "avg_match": self._calculate_avg_score([match]),
                "skills_verified": len(match.get("ranked_roles", []))
            }
        }
        
        # Format top 3 roles
        for i, role in enumerate(match.get("ranked_roles", [])[:3]):
            score = role.get("score", {})
            overlap = score.get("overlap", {})
            pct = overlap.get("percent_match", {})
            
            formatted["top_roles"].append({
                "rank": i + 1,
                "role_id": role["role_id"],
                "title": role.get("display_role_title") or role.get("role_title"),
                "match_score": {
                    "overall": pct.get("overall_pct", 0),
                    "must_have": pct.get("must_have_pct", 0),
                    "preferred": pct.get("preferred_pct", 0)
                },
                "matched_skills": overlap.get("keyword_match", {}).get("matched_must_haves", []),
                "missing_skills": overlap.get("keyword_mismatch", {}).get("missing_must_haves", []),
                "upskilling": self._extract_upskill_info(match, role["role_id"]),
                "manager_narrative": self._extract_manager_narrative(match, role["role_id"])
            })
        
        return formatted
    
    def _extract_upskill_info(self, match: Dict, role_id: str) -> Dict:
        """Extract upskilling plan for a specific role"""
        # Check if this is the top role with detailed plan
        plan = match.get("top_plan", {})
        
        if plan and match.get("ranked_roles", [{}])[0].get("role_id") == role_id:
            return {
                "available": True,
                "plan": plan.get("plan", []),
                "total_weeks": len(plan.get("plan", [])) * 2,
                "total_cost": sum(step.get("cost_estimate_usd", 0) 
                                for step in plan.get("plan", []))
            }
        
        return {"available": False}
    
    def _extract_manager_narrative(self, match: Dict, role_id: str) -> str:
        """Extract manager narrative for a specific role"""
        # Check if this is the top role with narrative
        if match.get("ranked_roles", [{}])[0].get("role_id") == role_id:
            return match.get("manager_narrative", "")
        return ""
    
    def _calculate_avg_score(self, matches: List[Dict]) -> float:
        """Calculate average match score across all matches"""
        if not matches:
            return 0.0
        
        total = 0
        count = 0
        
        for match in matches:
            for role in match.get("ranked_roles", []):
                score = role.get("score", {}).get("overall", 0)
                total += score * 100  # Convert to percentage
                count += 1
        
        return total / count if count > 0 else 0.0
    
    def _count_ready_candidates(self, matches: List[Dict], threshold: float = 0.75) -> int:
        """Count candidates ready to move (high match scores)"""
        count = 0
        for match in matches:
            if match.get("ranked_roles"):
                top_score = match["ranked_roles"][0].get("score", {}).get("overall", 0)
                if top_score >= threshold:
                    count += 1
        return count
    
    # ========================================================================
    # Demo mode (when engine not available)
    # ========================================================================
    
    def _get_demo_hr_results(self) -> Dict:
        """Generate demo results for HR dashboard"""
        return {
            "summary": {
                "total_candidates": 15,
                "total_roles": 8,
                "avg_match_score": 76.4,
                "ready_to_move": 6
            },
            "roles": {
                "R001": {
                    "role_id": "R001",
                    "title": "Senior Data Scientist",
                    "top_candidates": [
                        {
                            "employee_id": "E001",
                            "name": "Sarah Johnson",
                            "match_score": {"overall": 0.853, "must_have": 0.90, "preferred": 0.785},
                            "missing_skills": ["Deep Learning", "MLOps"],
                            "bias_severity": "none"
                        },
                        {
                            "employee_id": "E002",
                            "name": "Michael Chen",
                            "match_score": {"overall": 0.789, "must_have": 0.82, "preferred": 0.725},
                            "missing_skills": ["Python", "Statistical Modeling"],
                            "bias_severity": "low"
                        }
                    ]
                }
            },
            "errors": []
        }
    
    def _get_demo_employee_results(self, employee_id: str) -> Dict:
        """Generate demo results for employee view"""
        return {
            "employee_id": employee_id,
            "name": self._lookup_employee_name(employee_id),
            "top_roles": [
                {
                    "rank": 1,
                    "role_id": "R001",
                    "title": "Senior Data Scientist",
                    "match_score": {"overall": 85.3, "must_have": 90.0, "preferred": 78.5},
                    "matched_skills": ["Python", "SQL", "Statistics"],
                    "missing_skills": ["Deep Learning", "MLOps"],
                    "upskilling": {"available": True, "total_weeks": 8, "total_cost": 1200},
                    "manager_narrative": "Strong candidate with solid foundations..."
                }
            ],
            "career_metrics": {
                "avg_match": 82.0,
                "skills_verified": 18
            }
        }


# ============================================================================
# Convenience functions for direct use in dashboard
# ============================================================================

# Global instance
_integration = None

def get_integration(temp_dir: str = None, output_dir: str = None) -> DashboardIntegration:
    """Get or create global integration instance"""
    global _integration
    if _integration is None:
        _integration = DashboardIntegration(
            temp_dir=temp_dir or "data/dashboard_temp",
            output_dir=output_dir or "outputs/dashboard"
        )
    return _integration


def run_hr_analysis(resume_files, jd_files, config) -> Dict:
    """Convenience function for HR analysis"""
    integration = get_integration()
    return integration.process_hr_batch(resume_files, jd_files, config)


def run_employee_analysis(employee_id, resume=None, current_jd=None, 
                         target_jds=None, config=None) -> Dict:
    """Convenience function for employee analysis"""
    integration = get_integration()
    return integration.process_employee_analysis(
        employee_id, resume, current_jd, target_jds, config
    )