"""Command-line interface for Indian Address Parser."""

import argparse
import json
import sys
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Parse Indian addresses using NER",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse single address
  address-parser "PLOT NO752 FIRST FLOOR, NEW DELHI, 110041"

  # Parse from file
  address-parser --input addresses.txt --output parsed.json

  # Use trained model
  address-parser --model ./models/address_ner "H.NO. 123, LAJPAT NAGAR"
        """
    )

    parser.add_argument(
        "address",
        nargs="?",
        help="Address to parse (or use --input for file)"
    )
    parser.add_argument(
        "--input", "-i",
        help="Input file with addresses (one per line)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file"
    )
    parser.add_argument(
        "--model", "-m",
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "table", "simple"],
        default="json",
        help="Output format (default: json)"
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="indian-address-parser 2.0.0"
    )

    args = parser.parse_args()

    # Import here to avoid slow startup
    from address_parser import AddressParser

    # Load parser
    if args.model and Path(args.model).exists():
        print(f"Loading model from {args.model}...", file=sys.stderr)
        address_parser = AddressParser.from_pretrained(args.model)
    else:
        print("Using rules-only mode", file=sys.stderr)
        address_parser = AddressParser.rules_only()

    # Get addresses to parse
    addresses = []
    if args.input:
        with open(args.input, encoding="utf-8") as f:
            addresses = [line.strip() for line in f if line.strip()]
    elif args.address:
        addresses = [args.address]
    else:
        parser.print_help()
        sys.exit(1)

    # Parse addresses
    results = []
    for addr in addresses:
        result = address_parser.parse(addr)
        results.append(result)

    # Output
    if args.format == "json":
        output = [r.model_dump() for r in results]
        json_str = json.dumps(output, indent=2, ensure_ascii=False)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(json_str)
            print(f"Saved to {args.output}", file=sys.stderr)
        else:
            print(json_str)

    elif args.format == "table":
        for i, result in enumerate(results):
            print(f"\n{'='*60}")
            print(f"Address {i+1}: {result.raw_address[:50]}...")
            print(f"{'='*60}")
            print(f"{'Entity':<15} {'Value':<40} {'Conf':<6}")
            print("-" * 60)
            for entity in result.entities:
                print(f"{entity.label:<15} {entity.value:<40} {entity.confidence:.0%}")

    else:  # simple
        for result in results:
            parts = []
            if result.house_number:
                parts.append(f"House: {result.house_number}")
            if result.floor:
                parts.append(f"Floor: {result.floor}")
            if result.block:
                parts.append(f"Block: {result.block}")
            if result.gali:
                parts.append(f"Gali: {result.gali}")
            if result.colony:
                parts.append(f"Colony: {result.colony}")
            if result.area:
                parts.append(f"Area: {result.area}")
            if result.pincode:
                parts.append(f"PIN: {result.pincode}")
            if result.city:
                parts.append(f"City: {result.city}")

            print(" | ".join(parts) if parts else "No entities found")


if __name__ == "__main__":
    main()
