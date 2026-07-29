import os
import pandas as pd


def parse_protocol_file(protocol_path, audio_dir, output_csv_path):
    """Parses ASVspoof 2019 CM protocol ASCII text files and generates a

    structured CSV registry with sanity checks.
    """
    if not os.path.exists(protocol_path):
        print(f"❌ Protocol file not found at: {protocol_path}")
        return None

    if not os.path.exists(audio_dir):
        print(f"❌ Audio directory not found at: {audio_dir}")
        return None

    records = []
    missing_files = 0

    print(f"🔍 Parsing protocol: {os.path.basename(protocol_path)}...")

    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            speaker_id = parts[0]  # LA_****
            file_name = parts[1]  # LA_T_**** / LA_D_****
            system_id = parts[2]  # - (Bonafide) or A01-A19 (Spoof)
            key = parts[4]  # bonafide or spoof

            # Construct physical path (.flac format)
            audio_path = os.path.join(audio_dir, f"{file_name}.flac")

            # Verify physical file existence
            if not os.path.exists(audio_path):
                missing_files += 1
                continue

            label = 0 if key == "bonafide" else 1

            records.append(
                {
                    "speaker_id": speaker_id,
                    "file_name": file_name,
                    "audio_path": os.path.abspath(audio_path),
                    "system_id": system_id,
                    "key": key,
                    "label": label,
                }
            )

    df = pd.DataFrame(records)

    # Safeguard against empty results
    if df.empty:
        print(
            f"❌ Error: 0 valid audio files were matched inside: {os.path.abspath(audio_dir)}"
        )
        print(f"   • Total protocol entries checked: {missing_files}")
        print(
            f"   • Example missing file path: {os.path.abspath(os.path.join(audio_dir, 'LA_T_1000137.flac'))}"
        )
        print(
            "   👉 Please check if your .flac audio files are located directly in that folder.\n"
        )
        return df

    # Export to CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    # Dataset Summary Metrics
    print(f"✅ Successfully created: {output_csv_path}")
    print(f"   • Total Valid Audio Files: {len(df)}")
    print(
        f"   • Real (Bonafide) Samples: {len(df[df['label'] == 0])} ({len(df[df['label'] == 0])/len(df)*100:.1f}%)"
    )
    print(
        f"   • Fake (Spoof) Samples:    {len(df[df['label'] == 1])} ({len(df[df['label'] == 1])/len(df)*100:.1f}%)"
    )
    print(f"   • Unique Speakers:        {df['speaker_id'].nunique()}")
    if missing_files > 0:
        print(f"   ⚠️ Missing/Corrupt Files Skipped: {missing_files}")
    print("-" * 60)

    return df


if __name__ == "__main__":
    # Absolute path resolution based on repository root
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

    BASE_DIR = os.path.join(PROJECT_ROOT, "LA")

    # Protocol Files
    TRAIN_PROTOCOL = os.path.join(
        BASE_DIR,
        "ASVspoof2019_LA_cm_protocols",
        "ASVspoof2019.LA.cm.train.trn.txt",
    )
    DEV_PROTOCOL = os.path.join(
        BASE_DIR,
        "ASVspoof2019_LA_cm_protocols",
        "ASVspoof2019.LA.cm.dev.trl.txt",
    )

    # Audio Folders (Added 'flac' subfolder)
    TRAIN_AUDIO_DIR = os.path.join(BASE_DIR, "ASVspoof2019_LA_train", "flac")
    DEV_AUDIO_DIR = os.path.join(BASE_DIR, "ASVspoof2019_LA_dev", "flac")

    # Output CSV Destination
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data_registry")
    TRAIN_CSV = os.path.join(OUTPUT_DIR, "train_metadata.csv")
    DEV_CSV = os.path.join(OUTPUT_DIR, "dev_metadata.csv")

    print("🚀 Starting ASVspoof 2019 Dataset Organization...\n")
    train_df = parse_protocol_file(TRAIN_PROTOCOL, TRAIN_AUDIO_DIR, TRAIN_CSV)
    dev_df = parse_protocol_file(DEV_PROTOCOL, DEV_AUDIO_DIR, DEV_CSV)
    print("✨ Dataset arrangement complete!")
