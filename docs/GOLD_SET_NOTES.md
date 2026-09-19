# Gold set provenance

`configs/gold_set.yaml` is the labelled truth the evaluation harness scores
against. This file records **where each expectation came from**, so a reviewer
can challenge any label without re-reading 400 pages.

Every expectation was derived by opening the source documents directly, not by
running the extractor and copying its output (which would only prove the
extractor agrees with itself).

## Package `bid1` — Dallas ISD, RFP JA-207652

| Field | Expected | Source |
|---|---|---|
| `bid_number` | `JA-207652` | Master RFP p1: "Request For Proposal 168884 / JA-207652 Student and Staff Computing Devices"; BidNet "Solicitation Number" |
| `title` | contains "student and staff computing devices" | BidNet "Title" field |
| `due_date` | `2024-07-09T19:00:00Z` | **Addendum 2**: "The new due date for this RFP will be July 9, 2024 at 2:00 PM CST." Master RFP p2 header says `Solicitation Due 27-JUN-2024 14:00:00` — deliberately stale, so this checks supersession. |
| `bid_submission_type` | contains "electronic" | Master RFP p9: "Dallas ISD prefers responses to be submitted online via our electronic system" |
| `term_of_bid` | contains "three (3) year" | Master RFP p2: "The term of this proposal shall be for a three (3) year agreement with two (2) successive one (1) year extensions" |
| `pre_bid_meeting` | `2024-06-10T19:00:00Z` | Master RFP p2 header `Pre-Proposal Meeting 10-JUN-2024 14:00:00` (Central); BidNet "Prebid Conference" 06/10/2024 03:00 PM EDT. Same instant — checks instant-equality, not string equality. |
| `installation` | contains "software installation" | Master RFP p7: "services include (a) software installation, (b) asset tagging and etching…" |
| `bid_bond_requirement` | contains "bid bond" | Master RFP p9: "must comply with any insurance, bid bond, or liability requirements" |
| `delivery_date` | contains "september of 2024" | Addendum 1 Q4: "Dallas ISD is anticipating requesting quotes … starting in September of 2024" |
| `payment_terms` | absent | No payment-terms clause exists in any document. Asserting absence is deliberate: it penalises hallucination. |
| `additional_documentation_required` | contains "1295" | Addendum 1 Q31: "The Form 1295 should be submitted with the RFP response" |
| `mfg_for_registration` | contains "isupplier" | Master RFP p9: "Please register with the iSupplier portal" |
| `contract_or_cooperative_to_use` | absent | No cooperative/master contract is referenced by this solicitation. |
| `model_no`, `part_no`, `company_name` | absent | This is a **buyer-side** RFP. There is no vendor response, so no models, SKUs or bidder name exist. |
| `product` | contains "student and staff computing devices" | BidNet title |
| `contact_info` | contains "@dallasisd.org" | Buyer block p2: `JALZATE@dallasisd.org` |
| `bid_summary` | contains "request for proposal" | BidNet "Description" |
| `product_specification` | ≥ 5 entries | Master RFP pp4–6 list 8 device tiers ("Tier 1 Small Student Chromebook Laptop Minimum Requirements" …) |

## Package `bid2` — Maryland State Treasurer, BPM044557

| Field | Expected | Source |
|---|---|---|
| `bid_number` | `BPM044557` | PORFP p1: "eMMA Project Number: BPM044557"; BidNet "Solicitation Number" |
| `title` | contains "dell laptops" | BidNet "Title": "Dell Laptops w/Extended Warranty" |
| `due_date` | `2024-06-10T18:00:00Z` | PORFP p1 "PROPOSAL DUE DATE and TIME: 06/10/2024"; BidNet "Closing Date" 06/10/2024 02:00 PM EDT |
| `bid_submission_type` | contains "emaryland" | PORFP p1: "only be accepted through the State's eMaryland Marketplace Advantage (eMMA)" |
| `term_of_bid` | absent | The PORFP states no contract term. |
| `pre_bid_meeting`, `installation`, `bid_bond_requirement` | absent | Not stated in any document. |
| `delivery_date` | contains "45 days" | PORFP p2 item 9: "Delivery within 45 days of Award." |
| `payment_terms` | contains "invoice" | PORFP p2: "Invoice(s) shall be submitted within 10 days of delivering the equipment" |
| `additional_documentation_required` | contains "mercury affidavit" | PORFP p2 item 8 + the standalone `Mercury_Affidavit.pdf` |
| `mfg_for_registration` | contains "dell" | PORFP p2 item 4: "The Master Contractor must be an authorized reseller for Dell" |
| `contract_or_cooperative_to_use` | contains "master contract" | PORFP p1 heading "Hardware Master Contract"; p1 special instructions name "060B5400007" |
| `model_no` | contains "latitude 5550" | PORFP p3 Section 4 line items: "SI# CC7802 Dell Latitude 5550", "WD22TB4" |
| `part_no` | ≥ 20 items | `Dell_Laptop_Specs.pdf` pp1–2 list 34 Dell SKUs (210-BLYZ … 383-0464) |
| `company_name` | contains "dell" | `Dell_Laptop_Specs.pdf` footer: "Dell Marketing LP. U.S. only." |
| `contact_info` | contains "thawkins@treasurer.state.md.us" | PORFP p3: "Agency POC Name: Tamaira Hawkins", "410-260-7533" |
| `bid_summary` | contains "laptops" | PORFP p3 business need: "Office is in need of a refresh of laptops…" |
| `product_specification` | ≥ 2 entries | PORFP p3 Section 4 grid: 2 line items + 1 warranty entry |

## Deliberate design choices

- **Absence is scored.** Ten expectations assert a field must be `null`. A
  pipeline that invents a plausible value scores *worse* than one that returns
  nothing — this is the single most important property for procurement data.
- **Dates are compared by instant.** `2:00 PM CST` and `3:00 PM EDT` are one
  moment; comparing strings would fail a correct extraction.
- **`contains` over `equals` for prose.** Wording varies between sources; the
  requirement is that the fact is present, not a specific phrasing.
- **`min_items` for aggregates.** Counting 34 SKUs exactly would be brittle to
  layout noise; "at least 20" captures the real requirement (all SKUs found)
  without over-fitting.
