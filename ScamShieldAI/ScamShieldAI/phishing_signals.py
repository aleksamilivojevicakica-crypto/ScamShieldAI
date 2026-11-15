"""Interpretable, conservative phishing-risk signals; not a trained probability."""
import re

FINANCIAL = re.compile(r'\b(?:bank|banking|account|card|payment|wallet|credit|debit)\b', re.I)
THREAT = re.compile(r'\b(?:suspend(?:ed|sion)?|block(?:ed)?|lock(?:ed)?|restrict(?:ed|ion)?|frozen|freeze|deactivat(?:ed|ion)|unusual activity|unauthori[sz]ed)\b', re.I)
ACTION = re.compile(r'\b(?:verify|confirm|validate|update|restore|unlock|reactivate|provide|submit)\b', re.I)
IDENTITY = re.compile(r'\b(?:identity|personal information|password|passcode|pin|otp|one.time (?:password|code)|security code|login|credentials|details)\b', re.I)
URGENCY = re.compile(r'\b(?:urgent|immediately|right now|within \d+ (?:minutes|hours)|act now|final warning)\b', re.I)
LINK = re.compile(r'(?:https?://|www\.)\S+', re.I)


def assess_phishing(message):
    """Flag combined social-engineering patterns, without changing ML score."""
    patterns = [
        (FINANCIAL, 'Financial account or payment reference'),
        (THREAT, 'Threat of account restriction or suspicious activity'),
        (ACTION, 'Request to take a verification or recovery action'),
        (IDENTITY, 'Request involving identity or credentials'),
        (URGENCY, 'Urgent language'),
        (LINK, 'External link'),
    ]
    hits = [label for regex, label in patterns if regex.search(message)]
    financial, threat, action, identity, urgency, link = [bool(regex.search(message)) for regex, _ in patterns]
    # Require several mutually reinforcing signals to reduce false alarms.
    high_risk = financial and action and (identity or link) and (threat or urgency)
    elevated = financial and threat and action and urgency
    flagged = high_risk or elevated
    return {'flagged': flagged, 'signals': hits, 'level': 'High' if high_risk else 'Elevated' if elevated else 'Not triggered'}
