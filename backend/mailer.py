"""Ticket confirmation emails to the customer, sent through Brevo's HTTPS API.

HTTPS instead of SMTP because Render's free plan blocks the SMTP ports (25, 465, 587).
Brevo can send to any address once the sender address is verified (no domain needed).
Without BREVO_API_KEY and MAIL_FROM, the email is only logged, as before.
"""
import json
import logging
import os
import threading
import urllib.error
import urllib.request

log = logging.getLogger("support")
BREVO_URL = "https://api.brevo.com/v3/smtp/email"


def send_ticket_email(ticket: dict) -> None:
    """Email the customer their ticket number. Never raises: a failed email can't break the ticket."""
    subject = f"Your support ticket #{ticket['id']} has been created"
    body = (f"Hi {ticket['customer_name']},\n\n"
            f"We've received your request and created support ticket #{ticket['id']}.\n\n"
            f"Summary: {ticket['summary']}\n"
            f"Category: {ticket['category']}\n"
            f"Status: {ticket['status']}\n\n"
            "Our support team will follow up with you at this email address.\n\n"
            "Customer Support")
    api_key, sender = os.getenv("BREVO_API_KEY", "").strip(), os.getenv("MAIL_FROM", "").strip()
    if not (api_key and sender):
        log.info("[SIMULATED EMAIL] To: %s | %s | %s", ticket["customer_email"], subject, ticket["summary"])
        return
    payload = {
        "sender": {"name": "Customer Support", "email": sender},
        "to": [{"email": ticket["customer_email"], "name": ticket["customer_name"]}],
        "subject": subject,
        "textContent": body,  # plain text, so nothing the customer typed can inject HTML
    }
    # A background thread, so the chat reply isn't held up by the email API.
    # ponytail: fire-and-forget, no retry; use a job queue if an email must never be lost.
    threading.Thread(target=_post, args=(api_key, payload, ticket["id"]), daemon=True).start()


def _post(api_key: str, payload: dict, ticket_id: int) -> None:
    request = urllib.request.Request(
        BREVO_URL, data=json.dumps(payload).encode(),
        headers={"api-key": api_key, "content-type": "application/json", "accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15):
            log.info("Emailed ticket #%s confirmation to %s", ticket_id, payload["to"][0]["email"])
    except urllib.error.HTTPError as e:  # Brevo explains the problem in the body (bad key, unverified sender…)
        log.warning("Ticket #%s email failed: HTTP %s %s", ticket_id, e.code, e.read().decode(errors="replace")[:300])
    except OSError as e:  # network error or timeout
        log.warning("Ticket #%s email failed: %s", ticket_id, e)
