import json
import os
import re

def safe_filename(text: str, max_len: int = 80) -> str:
    """Convert a topic name (or meeting ID) to a safe filename."""
    # Remove invalid characters
    safe = re.sub(r'[<>:"/\\|?*]', '_', text)
    # Remove extra spaces and truncate
    safe = re.sub(r'\s+', ' ', safe).strip()
    if len(safe) > max_len:
        safe = safe[:max_len]
    return safe

def split_meetingbank_json(input_file: str, output_dir: str, split_by: str = "topic"):
    """
    Split the MeetingBank JSON into smaller JSON files.

    Args:
        input_file: Path to the input JSON file (containing "MeetingBank" object)
        output_dir: Directory where split files will be saved
        split_by: "topic" (default) or "meeting" – determines granularity
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # The root might be "MeetingBank" or directly the meetings object
    if "MeetingBank" in data:
        meetings_dict = data["MeetingBank"]
    else:
        meetings_dict = data  # assume it's already the meetings dict

    os.makedirs(output_dir, exist_ok=True)

    for meeting_id, meeting_data in meetings_dict.items():
        topics = meeting_data.get("topics", {})
        
        if split_by == "meeting":
            # One JSON file per meeting (contains all topics)
            out_file = os.path.join(output_dir, f"{safe_filename(meeting_id)}.json")
            with open(out_file, 'w', encoding='utf-8') as out_f:
                json.dump({meeting_id: meeting_data}, out_f, indent=2)
            print(f"Saved meeting: {out_file}")
        else:  # split_by == "topic"
            # One JSON file per topic
            for topic_name, topic_content in topics.items():
                # Create a safe filename from meeting_id + topic_name
                base_name = f"{meeting_id}_{safe_filename(topic_name, max_len=100)}"
                out_file = os.path.join(output_dir, f"{base_name}.json")
                # Save only this topic's content (preserve structure or simplify)
                topic_data = {
                    "meeting_id": meeting_id,
                    "topic_name": topic_name,
                    **topic_content
                }
                with open(out_file, 'w', encoding='utf-8') as out_f:
                    json.dump(topic_data, out_f, indent=2)
                print(f"Saved topic: {out_file}")

    print(f"\n✅ Splitting complete. Files saved to {output_dir}")

# Example usage
if __name__ == "__main__":
    input_json = "meetingbank_data.json"   # your big JSON file
    output_directory = "datasplit_topics"  # where your RAG pipeline expects JSONs
    split_meetingbank_json(input_json, output_directory, split_by="topic")