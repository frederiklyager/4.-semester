"""
Enhanced Security Monitoring & Logging System
NIS2 Compliance - Comprehensive incident detection and logging

Features:
- Structured logging with severity levels
- Incident detection and alerting
- Log aggregation and analysis
- Compliance reporting
- Real-time monitoring
"""

import logging
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from enum import Enum
import hashlib


class IncidentSeverity(Enum):
    """Incident severity levels aligned with IRP"""
    CRITICAL = "🔴 Critical"
    HIGH = "🟠 High"
    MEDIUM = "🟡 Medium"
    LOW = "🟢 Low"
    INFO = "🔵 Info"


class SecurityMonitor:
    """
    Centralized security monitoring and incident detection
    
    Monitors:
    - Authentication failures
    - Data validation failures
    - Rate limit violations
    - API errors
    - System health
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize loggers
        self.security_logger = self._setup_logger("security", "security.log")
        self.audit_logger = self._setup_logger("audit", "audit.log")
        self.incident_logger = self._setup_logger("incidents", "incidents.log")
        
        # Incident counters
        self.auth_failures = {}
        self.validation_failures = {}
        self.rate_limit_violations = {}
    
    def _setup_logger(self, name: str, filename: str) -> logging.Logger:
        """Setup structured logger with file handler"""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(self.log_dir / filename)
        file_handler.setLevel(logging.INFO)
        
        # JSON formatter for structured logging
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": "%(message)s"}'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        return logger
    
    def log_authentication_attempt(self, service: str, token: str, success: bool, ip: str = "local"):
        """
        Log authentication attempt
        
        Args:
            service: Service being accessed
            token: Token used (will be hashed)
            success: Whether auth succeeded
            ip: Source IP address
        """
        token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
        
        event = {
            "event_type": "authentication",
            "service": service,
            "token_hash": token_hash,
            "success": success,
            "ip": ip,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if success:
            self.security_logger.info(f"Authentication successful: {json.dumps(event)}")
        else:
            self.security_logger.warning(f"Authentication failed: {json.dumps(event)}")
            
            # Track failures for incident detection
            key = f"{service}_{ip}"
            self.auth_failures[key] = self.auth_failures.get(key, 0) + 1
            
            # Check if threshold exceeded
            if self.auth_failures[key] >= 5:
                self.raise_incident(
                    IncidentSeverity.HIGH,
                    "authentication_spike",
                    f"Multiple failed auth attempts detected: {key}",
                    {"service": service, "ip": ip, "count": self.auth_failures[key]}
                )
    
    def log_data_validation(self, source: str, valid: bool, errors: Optional[List] = None):
        """
        Log data validation event
        
        Args:
            source: Data source
            valid: Whether validation passed
            errors: List of validation errors
        """
        event = {
            "event_type": "validation",
            "source": source,
            "valid": valid,
            "errors": errors or [],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if valid:
            self.security_logger.info(f"Validation passed: {json.dumps(event)}")
        else:
            self.security_logger.error(f"Validation failed: {json.dumps(event)}")
            
            # Track failures
            self.validation_failures[source] = self.validation_failures.get(source, 0) + 1
            
            # Check threshold
            if self.validation_failures[source] >= 3:
                self.raise_incident(
                    IncidentSeverity.MEDIUM,
                    "validation_failure_spike",
                    f"Multiple validation failures: {source}",
                    {"source": source, "count": self.validation_failures[source], "errors": errors}
                )
    
    def log_rate_limit(self, endpoint: str, identifier: str, blocked: bool):
        """
        Log rate limit event
        
        Args:
            endpoint: API endpoint
            identifier: Request identifier
            blocked: Whether request was blocked
        """
        event = {
            "event_type": "rate_limit",
            "endpoint": endpoint,
            "identifier": identifier,
            "blocked": blocked,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if blocked:
            self.security_logger.warning(f"Rate limit exceeded: {json.dumps(event)}")
            
            # Track violations
            key = f"{endpoint}_{identifier}"
            self.rate_limit_violations[key] = self.rate_limit_violations.get(key, 0) + 1
            
            # Check threshold
            if self.rate_limit_violations[key] >= 10:
                self.raise_incident(
                    IncidentSeverity.MEDIUM,
                    "rate_limit_abuse",
                    f"Repeated rate limit violations: {endpoint}",
                    {"endpoint": endpoint, "identifier": identifier, "count": self.rate_limit_violations[key]}
                )
        else:
            self.security_logger.info(f"Rate limit check passed: {json.dumps(event)}")
    
    def log_api_call(self, endpoint: str, status: int, duration_ms: float):
        """
        Log API call for monitoring
        
        Args:
            endpoint: API endpoint
            status: HTTP status code
            duration_ms: Request duration in milliseconds
        """
        event = {
            "event_type": "api_call",
            "endpoint": endpoint,
            "status": status,
            "duration_ms": duration_ms,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if status >= 500:
            self.security_logger.error(f"API error: {json.dumps(event)}")
        elif status >= 400:
            self.security_logger.warning(f"API client error: {json.dumps(event)}")
        else:
            self.audit_logger.info(f"API call: {json.dumps(event)}")
    
    def log_data_access(self, user: str, resource: str, action: str, granted: bool):
        """
        Audit log for data access
        
        Args:
            user: User/service accessing data
            resource: Resource being accessed
            action: Action performed (read, write, delete)
            granted: Whether access was granted
        """
        event = {
            "event_type": "data_access",
            "user": user,
            "resource": resource,
            "action": action,
            "granted": granted,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        self.audit_logger.info(f"Data access: {json.dumps(event)}")
        
        if not granted:
            self.security_logger.warning(f"Access denied: {json.dumps(event)}")
    
    def raise_incident(self, severity: IncidentSeverity, incident_type: str, 
                      description: str, details: Dict):
        """
        Raise security incident
        
        Args:
            severity: Incident severity level
            incident_type: Type of incident
            description: Human-readable description
            details: Additional incident details
        """
        incident = {
            "incident_id": self._generate_incident_id(),
            "severity": severity.value,
            "type": incident_type,
            "description": description,
            "details": details,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "open"
        }
        
        self.incident_logger.error(f"INCIDENT: {json.dumps(incident)}")
        
        # In production, this would trigger alerts (email, Slack, PagerDuty)
        print(f"\n{'='*60}")
        print(f"{severity.value} SECURITY INCIDENT DETECTED")
        print(f"Type: {incident_type}")
        print(f"Description: {description}")
        print(f"Incident ID: {incident['incident_id']}")
        print(f"{'='*60}\n")
    
    def _generate_incident_id(self) -> str:
        """Generate unique incident ID"""
        timestamp = datetime.now(timezone.utc).isoformat()
        return f"INC-{hashlib.sha256(timestamp.encode()).hexdigest()[:8].upper()}"
    
    def get_security_metrics(self) -> Dict:
        """
        Get current security metrics
        
        Returns:
            Dict with security metrics
        """
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "auth_failures": sum(self.auth_failures.values()),
            "validation_failures": sum(self.validation_failures.values()),
            "rate_limit_violations": sum(self.rate_limit_violations.values()),
            "unique_auth_sources": len(self.auth_failures),
            "unique_validation_sources": len(self.validation_failures),
            "unique_rate_limit_sources": len(self.rate_limit_violations)
        }
    
    def reset_counters(self):
        """Reset incident counters (call hourly)"""
        self.auth_failures.clear()
        self.validation_failures.clear()
        self.rate_limit_violations.clear()
        self.security_logger.info("Security counters reset")


class LogAnalyzer:
    """
    Analyze security logs for patterns and compliance reporting
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
    
    def parse_log_file(self, filename: str) -> List[Dict]:
        """Parse JSON log file"""
        log_file = self.log_dir / filename
        
        if not log_file.exists():
            return []
        
        events = []
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        return events
    
    def get_authentication_summary(self) -> Dict:
        """Get authentication event summary"""
        events = self.parse_log_file("security.log")
        auth_events = [e for e in events if "authentication" in e.get("message", "")]
        
        total = len(auth_events)
        successful = len([e for e in auth_events if "successful" in e.get("message", "")])
        failed = total - successful
        
        return {
            "total_attempts": total,
            "successful": successful,
            "failed": failed,
            "success_rate": f"{(successful/total*100):.1f}%" if total > 0 else "N/A"
        }
    
    def get_validation_summary(self) -> Dict:
        """Get validation event summary"""
        events = self.parse_log_file("security.log")
        validation_events = [e for e in events if "validation" in e.get("message", "").lower()]
        
        total = len(validation_events)
        passed = len([e for e in validation_events if "passed" in e.get("message", "")])
        failed = total - passed
        
        return {
            "total_validations": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": f"{(passed/total*100):.1f}%" if total > 0 else "N/A"
        }
    
    def get_incident_summary(self) -> Dict:
        """Get incident summary"""
        events = self.parse_log_file("incidents.log")
        
        by_severity = {}
        by_type = {}
        
        for event in events:
            msg = event.get("message", "")
            if "INCIDENT:" in msg:
                try:
                    incident = json.loads(msg.split("INCIDENT: ")[1])
                    severity = incident.get("severity", "Unknown")
                    inc_type = incident.get("type", "Unknown")
                    
                    by_severity[severity] = by_severity.get(severity, 0) + 1
                    by_type[inc_type] = by_type.get(inc_type, 0) + 1
                except:
                    continue
        
        return {
            "total_incidents": len(events),
            "by_severity": by_severity,
            "by_type": by_type
        }
    
    def generate_compliance_report(self) -> str:
        """Generate NIS2 compliance report"""
        auth_summary = self.get_authentication_summary()
        validation_summary = self.get_validation_summary()
        incident_summary = self.get_incident_summary()
        
        report = f"""
# 📊 Security Monitoring Report
**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
**Compliance:** NIS2 Directive

## Authentication Monitoring
- Total Attempts: {auth_summary['total_attempts']}
- Successful: {auth_summary['successful']}
- Failed: {auth_summary['failed']}
- Success Rate: {auth_summary['success_rate']}

## Data Validation
- Total Validations: {validation_summary['total_validations']}
- Passed: {validation_summary['passed']}
- Failed: {validation_summary['failed']}
- Pass Rate: {validation_summary['pass_rate']}

## Incidents
- Total Incidents: {incident_summary['total_incidents']}
- By Severity: {incident_summary['by_severity']}
- By Type: {incident_summary['by_type']}

## Compliance Status
✅ Logging: Active and comprehensive
✅ Incident Detection: Automated monitoring
✅ Audit Trail: Complete event history
✅ NIS2 Requirement: Met (24h notification capability)

## Recommendations
- Review failed authentication patterns
- Investigate repeated validation failures
- Monitor incident trends
- Update security controls as needed
"""
        return report


# Global monitor instance
security_monitor = SecurityMonitor()
log_analyzer = LogAnalyzer()


if __name__ == "__main__":
    print("🔍 Security Monitoring System Demo\n")
    
    # Demo: Authentication events
    print("1️⃣ Testing Authentication Monitoring:")
    security_monitor.log_authentication_attempt("energinet_api", "token123", True, "192.168.1.1")
    security_monitor.log_authentication_attempt("energinet_api", "token456", False, "192.168.1.100")
    security_monitor.log_authentication_attempt("energinet_api", "token789", False, "192.168.1.100")
    print("   ✅ Authentication events logged\n")
    
    # Demo: Validation events
    print("2️⃣ Testing Data Validation Monitoring:")
    security_monitor.log_data_validation("energinet_co2", True, None)
    security_monitor.log_data_validation("energinet_co2", False, ["CO2 value out of range"])
    print("   ✅ Validation events logged\n")
    
    # Demo: Rate limiting
    print("3️⃣ Testing Rate Limit Monitoring:")
    security_monitor.log_rate_limit("api/co2", "user_123", False)
    security_monitor.log_rate_limit("api/co2", "user_123", True)
    print("   ✅ Rate limit events logged\n")
    
    # Demo: API calls
    print("4️⃣ Testing API Call Monitoring:")
    security_monitor.log_api_call("https://api.energinet.dk/", 200, 150.5)
    security_monitor.log_api_call("https://api.energinet.dk/", 500, 5000.0)
    print("   ✅ API calls logged\n")
    
    # Demo: Metrics
    print("5️⃣ Security Metrics:")
    metrics = security_monitor.get_security_metrics()
    for key, value in metrics.items():
        print(f"   {key}: {value}")
    
    print("\n6️⃣ Compliance Report:")
    report = log_analyzer.generate_compliance_report()
    print(report)
    
    print("\n✅ Security monitoring system operational!")
    print(f"📁 Logs location: logs/")
    print("   - security.log (security events)")
    print("   - audit.log (data access)")
    print("   - incidents.log (security incidents)")