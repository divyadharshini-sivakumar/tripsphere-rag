"""
Generate realistic sample documents for TripSphere:
- PDF  : hotel policies
- CSV  : booking & pricing data
- TXT  : travel-support FAQs
"""
from pathlib import Path
import csv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

from config import DATA_DIR

DATA_DIR.mkdir(exist_ok=True)


def create_hotel_policy_pdf():
    path = DATA_DIR / "tripsphere_hotel_policies.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=letter)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleCustom", parent=styles["Heading1"], fontSize=16, spaceAfter=12
    )
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, spaceBefore=10)
    body = styles["BodyText"]

    story = []
    story.append(Paragraph("TripSphere Hotel Policies Handbook", title_style))
    story.append(Paragraph("Effective Date: January 1, 2026", body))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("1. Check-in & Check-out", h2))
    story.append(
        Paragraph(
            "Standard check-in time is 3:00 PM local time. Early check-in is subject to "
            "availability and may incur a fee of $25–$50. Check-out is at 11:00 AM. "
            "Late check-out until 2:00 PM can be requested and is complimentary for Gold "
            "and Platinum members; otherwise a half-day rate may apply.",
            body,
        )
    )

    story.append(Paragraph("2. Cancellation Policy", h2))
    story.append(
        Paragraph(
            "Flexible Rate: Free cancellation up to 24 hours before arrival. "
            "Non-Refundable Rate: No refunds after booking confirmation. "
            "Group bookings (10+ rooms) require 14-day notice for full refund. "
            "In case of force majeure (natural disasters, government travel bans), "
            "TripSphere will offer a full credit valid for 12 months.",
            body,
        )
    )

    story.append(Paragraph("3. Pet Policy", h2))
    story.append(
        Paragraph(
            "Pets are welcome at select TripSphere properties. A non-refundable pet fee "
            "of $50 per stay applies. Maximum two pets per room, each under 40 lbs. "
            "Service animals are exempt from fees and weight limits. Pets must be "
            "leashed in public areas. Owners are responsible for any damage.",
            body,
        )
    )

    story.append(Paragraph("4. Loyalty Program – Sphere Rewards", h2))
    story.append(
        Paragraph(
            "Silver (0–9 nights/year): 5% discount on future stays. "
            "Gold (10–24 nights): 10% discount + free late check-out. "
            "Platinum (25+ nights): 15% discount, room upgrades when available, "
            "and complimentary breakfast. Points never expire for active members.",
            body,
        )
    )

    story.append(Paragraph("5. Smoking & Quiet Hours", h2))
    story.append(
        Paragraph(
            "All TripSphere hotels are 100% non-smoking indoors. Designated outdoor "
            "smoking areas are provided. Quiet hours are 10:00 PM – 7:00 AM. "
            "Violations may result in a $250 cleaning fee or eviction without refund.",
            body,
        )
    )

    story.append(Paragraph("6. Payment & Deposit", h2))
    story.append(
        Paragraph(
            "A valid credit card is required at check-in for incidentals. "
            "A hold of $100–$200 per night may be placed. Final settlement occurs "
            "at check-out. We accept Visa, Mastercard, Amex, and TripSphere Gift Cards. "
            "Corporate accounts may use direct billing with prior approval.",
            body,
        )
    )

    doc.build(story)
    print(f"Created {path}")


def create_booking_csv():
    path = DATA_DIR / "tripsphere_bookings_pricing.csv"
    headers = [
        "booking_id",
        "hotel_name",
        "city",
        "room_type",
        "price_per_night_usd",
        "availability",
        "max_occupancy",
        "amenities",
        "season",
    ]
    rows = [
        ["TS-1001", "Sphere Grand Downtown", "New York", "Standard King", 189, "Available", 2, "WiFi,Gym,Breakfast", "Low"],
        ["TS-1002", "Sphere Grand Downtown", "New York", "Deluxe Suite", 349, "Limited", 3, "WiFi,Gym,Breakfast,Spa", "Low"],
        ["TS-1003", "Ocean Sphere Resort", "Miami", "Ocean View Queen", 229, "Available", 2, "WiFi,Pool,Beach Access", "High"],
        ["TS-1004", "Ocean Sphere Resort", "Miami", "Family Villa", 459, "Available", 6, "WiFi,Pool,Kitchen,Beach", "High"],
        ["TS-1005", "Mountain Sphere Lodge", "Denver", "Standard Twin", 159, "Available", 2, "WiFi,Fireplace,Ski Storage", "Low"],
        ["TS-1006", "Mountain Sphere Lodge", "Denver", "Fireplace Suite", 279, "Limited", 4, "WiFi,Fireplace,Hot Tub", "Low"],
        ["TS-1007", "Sphere Airport Hub", "Chicago", "Standard Queen", 129, "Available", 2, "WiFi,Shuttle,Gym", "Shoulder"],
        ["TS-1008", "Sphere Airport Hub", "Chicago", "Business King", 179, "Available", 2, "WiFi,Shuttle,Desk,Breakfast", "Shoulder"],
        ["TS-1009", "Desert Sphere Oasis", "Phoenix", "Casita", 199, "Available", 3, "WiFi,Pool,Patio", "High"],
        ["TS-1010", "Desert Sphere Oasis", "Phoenix", "Premium Suite", 329, "Limited", 4, "WiFi,Pool,Spa,Patio", "High"],
        ["TS-1011", "Sphere Grand Downtown", "New York", "Penthouse", 799, "Request", 4, "WiFi,Gym,Spa,Butler", "Low"],
        ["TS-1012", "Ocean Sphere Resort", "Miami", "Standard King", 189, "Available", 2, "WiFi,Pool", "High"],
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"Created {path}")


def create_faq_txt():
    path = DATA_DIR / "tripsphere_travel_faqs.txt"
    content = """TripSphere Travel Support FAQs
Last updated: March 2026

Q: How do I modify or cancel an existing reservation?
A: Log into your TripSphere account, go to My Bookings, select the reservation, and choose Modify or Cancel. Flexible rates allow free cancellation up to 24 hours before check-in. Non-refundable rates cannot be cancelled for a refund but may be changed for a fee.

Q: What is the Sphere Rewards loyalty program?
A: Sphere Rewards is TripSphere's free loyalty program. Earn points on every paid night. Tiers are Silver, Gold, and Platinum based on nights stayed in a calendar year. Benefits include discounts, free upgrades, late check-out, and complimentary breakfast at higher tiers.

Q: Are there airport shuttle services?
A: Yes. Sphere Airport Hub (Chicago) offers a complimentary 24/7 shuttle. Ocean Sphere Resort (Miami) provides paid shared shuttles for $25 per person. Other properties can arrange private transfers through the concierge for an additional fee.

Q: Can I use points for free nights?
A: Yes. Gold and Platinum members can redeem Sphere Points for free nights. Redemption rates start at 8,000 points per night for standard rooms. Points plus cash options are also available.

Q: What is the policy for lost items?
A: Contact the hotel front desk within 7 days of check-out. TripSphere will hold lost items for 30 days. Shipping fees for returned items are the guest's responsibility unless the loss was caused by hotel staff.

Q: Do you offer accessible rooms?
A: All TripSphere properties have ADA-compliant rooms available. Features include roll-in showers, grab bars, lowered fixtures, and visual alarms. Request accessible rooms at the time of booking or by calling 1-800-TRIPSPHERE.

Q: How does dynamic pricing work?
A: Prices fluctuate based on demand, season, local events, and remaining inventory. Booking early or during low season usually yields the best rates. Sphere Rewards members receive member-only rates that are often lower than public rates.

Q: Is breakfast included?
A: Breakfast is included for Platinum members at all properties and for Gold members at select locations. Other guests can purchase breakfast packages at the front desk or add them during booking for $18–$28 per person per day.

Q: What payment methods are accepted?
A: We accept major credit cards (Visa, Mastercard, American Express), debit cards, TripSphere Gift Cards, and in some locations Apple Pay / Google Pay. Corporate direct billing is available with an approved account.

Q: How can I contact TripSphere support?
A: 24/7 support is available at 1-800-TRIPSPHERE or support@tripsphere.com. Live chat is available in the mobile app and on the website between 6 AM and 12 AM ET.
"""
    path.write_text(content, encoding="utf-8")
    print(f"Created {path}")


if __name__ == "__main__":
    print("Generating TripSphere sample documents...")
    create_hotel_policy_pdf()
    create_booking_csv()
    create_faq_txt()
    print("Done. Files are in the data/ directory.")
