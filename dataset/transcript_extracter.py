def format_conversations(conversations):
    output_data = []
    for conv in conversations:
        # Get speaker and gender from the first message
        speaker = conv['transcript'][0].get(
            'speaker', 'unknown') if conv.get('transcript') else 'unknown'
        gender = conv['transcript'][0].get(
            'gender', 'unknown') if conv.get('transcript') else 'unknown'

        # Concatenate messages into a single string
        conversation_text = " ".join(
            message.get('message', '')
            for message in conv.get('transcript', []))

        # Create the new JSON object
        output_data.append({
            "type": conv.get('type', ''),
            "category": conv.get('category', ''),
            "speaker": speaker,
            "gender": gender,
            "conversation": conversation_text
        })

    return output_data


# Extract conversations based on type, category, and speaker filters
def extract_conversations(data, type_filter, category_filter, speaker_filter):
    filtered_conversations = []

    for conversation in data:
        # Check if conversation matches type filter (if provided)
        if type_filter and conversation.get('type') != type_filter:
            continue

        # Check if conversation matches category filter (if provided)
        if category_filter and conversation.get('category') != category_filter:
            continue

        # If speaker filter is provided, only include messages from that speaker
        if speaker_filter:
            filtered_transcript = [
                message for message in conversation.get('transcript', [])
                if message.get('speaker') == speaker_filter
            ]
            if filtered_transcript:  # Only include conversation if there are matching messages
                filtered_conversation = conversation.copy()
                filtered_conversation['transcript'] = filtered_transcript
                filtered_conversations.append(filtered_conversation)
        else:
            filtered_conversations.append(conversation)

    return format_conversations(filtered_conversations)
