"""
Demonstration: Live Session Mode with UncertaintyAuditor

This example shows how to:
1. Set up a live session with mock audio
2. Register the UncertaintyAuditor agent
3. Monitor transcription confidence in real-time
4. Review flagged segments after the session
"""

import asyncio
from datetime import datetime
from pacing import (
    PacingPlatform,
    OperatingMode,
    SessionMetadata,
    MockAudioProvider,
    AdaptiveConfidenceTranscriber,
    UncertaintyAuditor,
)


async def main():
    print("=" * 70)
    print("PACING Demo: Live Session with Uncertainty Auditing")
    print("=" * 70)

    # Initialize platform in DEV mode (for demonstration purposes)
    print("\n🔧 Initializing platform in DEV_MODE...")
    platform = PacingPlatform(operating_mode=OperatingMode.DEV_MODE)

    # Use AdaptiveConfidenceTranscriber (generates realistic low-confidence segments)
    transcriber = AdaptiveConfidenceTranscriber()
    platform.set_transcriber(transcriber)

    # Register UncertaintyAuditor
    print("🔍 Registering UncertaintyAuditor (threshold: 0.70)...")
    auditor = UncertaintyAuditor(
        confidence_threshold=0.70, auto_flag_medical_terms=True
    )
    platform.register_agent(auditor)

    # Create session metadata
    session = SessionMetadata(
        session_id="demo_session_001",
        patient_id="patient_123",
        clinician_id="clinician_456",
        start_time=datetime.now(),
        session_type="counseling",
    )

    print(f"\n📋 Session Information:")
    print(f"   Session ID: {session.session_id}")
    print(f"   Patient ID: {session.patient_id}")
    print(f"   Clinician ID: {session.clinician_id}")

    # Create mock audio provider (30 seconds of scripted conversation)
    print("\n🎤 Starting audio capture (mock, 30 seconds)...")
    audio = MockAudioProvider(total_duration_sec=30.0)

    # Start the session
    print("\n▶️  SESSION STARTED")
    print("-" * 70)
    print("Transcribing and monitoring confidence in real-time...\n")

    await platform.start_live_session(session, audio)

    # Wait for session to complete
    # In a real application, this would run until manually stopped
    await asyncio.sleep(35)

    # Stop the session
    await platform.stop_live_session()

    print("\n⏹️  SESSION ENDED")
    print("=" * 70)

    # Review flagged segments
    print("\n📊 AUDIT REPORT")
    print("-" * 70)

    review_queue = auditor.get_review_queue()
    stats = auditor.get_statistics()

    print(f"\nTranscriptions Processed: {stats['session_stats']['transcriptions_processed']}")
    print(f"Segments Flagged: {stats['session_stats']['transcriptions_flagged']}")
    print(f"Flagging Rate: {stats['session_stats']['flagging_rate']:.1%}")

    if review_queue:
        print(f"\n🚨 FLAGGED SEGMENTS REQUIRING REVIEW ({len(review_queue)} items):")
        print("-" * 70)

        for i, item in enumerate(review_queue, 1):
            priority_stars = "⭐" * item.priority
            confidence = item.transcription.confidence_score

            print(f"\n{i}. Priority: {item.priority}/5 {priority_stars}")
            print(f"   Text: \"{item.transcription.text}\"")
            print(f"   Confidence: {confidence:.1%}")
            print(f"   Reason: {item.reason}")
            print(f"   Timestamp: {item.flagged_at.strftime('%H:%M:%S')}")

        print("\n" + "=" * 70)
        print("RECOMMENDED ACTIONS")
        print("=" * 70)
        print(
            """
1. Review high-priority items (4-5 stars) immediately
2. Verify medical terms for accuracy (medication names, dosages)
3. Re-listen to audio for low-confidence segments if available
4. Correct any errors in the clinical documentation
5. Consider flagging patterns: Are certain terms consistently low-confidence?

In a production system, these would be:
- Sent to a review queue for human verification
- Potentially re-transcribed with alternative methods
- Annotated with corrected text by clinical staff
        """
        )

        # Demonstrate marking items as reviewed
        print("\n🔄 DEMO: Marking first item as reviewed...")
        if review_queue:
            first_item = review_queue[0]
            auditor.mark_reviewed(
                first_item.item_id,
                reviewer_notes="Verified correct - 'buprenorphine' was accurately transcribed",
            )
            print(f"✓ Item {first_item.item_id} marked as reviewed")

            unreviewed = auditor.get_unreviewed_items()
            print(f"\nUnreviewed items remaining: {len(unreviewed)}")

    else:
        print("\n✅ No segments flagged - all transcriptions met confidence threshold!")

    print("\n" + "=" * 70)
    print("Platform Status:")
    print("=" * 70)
    status = platform.get_platform_status()
    print(f"Operating Mode: {status['operating_mode'].upper()}")
    print(f"Session Active: {status['session_active']}")
    print(f"Transcriber: {status['transcriber']}")
    print(f"Registered Agents: {', '.join(status['registered_agents'])}")

    print("\n" + "=" * 70)
    print("Technical Notes:")
    print("=" * 70)
    print(
        """
• DEV_MODE: Transcripts are persisted for auditing and debugging
• PROD_MODE: Transcripts would be ephemeral (only flagged items retained)
• The UncertaintyAuditor provides a critical safety mechanism for clinical AI
• Human verification ensures that clinical decisions are based on accurate data
• This "glass box" approach makes AI explainable and auditable
"""
    )


if __name__ == "__main__":
    asyncio.run(main())
