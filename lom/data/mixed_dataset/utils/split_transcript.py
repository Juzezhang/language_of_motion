# import textgrid as tg

def split_and_merge_sentences(intervals, max_duration=10.0):
    # List to store all identified sentences
    sentences = []
    # List to accumulate words in the current sentence
    current_sentence = []
    # Variables to track the start and end times of the current sentence
    current_sentence_start_time = None
    current_sentence_end_time = None

    # Step 1: Split sentences based on 'None' (pause) markers and max_duration limit
    for interval in intervals:
        # Get the word and its start and end times from the interval
        word = interval.mark
        word_start_time = interval.minTime * 1000  # Convert to milliseconds
        word_end_time = interval.maxTime * 1000

        # Check if the interval is a pause (marked as 'None' or whitespace)
        if not word or word.isspace():
            # Check if current sentence duration exceeds max_duration
            if current_sentence and (current_sentence_end_time - current_sentence_start_time) / 1000.0 > max_duration:
                # If sentence is too long, split into smaller sentences within max_duration
                split_and_add_sentences(current_sentence, current_sentence_start_time, current_sentence_end_time, sentences, max_duration)
                current_sentence = []
                current_sentence_start_time = None
                current_sentence_end_time = None

            # If there are words in the current sentence, save it as a complete sentence
            elif current_sentence:
                sentences.append((current_sentence, current_sentence_start_time, current_sentence_end_time))
                current_sentence = []  # Clear the current sentence
                current_sentence_start_time = None
                current_sentence_end_time = None
        else:
            # Accumulate words and update the current sentence times
            if not current_sentence:
                # Initialize the start time of the current sentence
                current_sentence_start_time = word_start_time
            # Add the current word to the sentence
            current_sentence.append(word)
            # Update the end time of the current sentence
            current_sentence_end_time = word_end_time

    # Handle the last sentence (if present)
    if current_sentence:
        if (current_sentence_end_time - current_sentence_start_time) / 1000.0 > max_duration:
            split_and_add_sentences(current_sentence, current_sentence_start_time, current_sentence_end_time, sentences, max_duration)
        else:
            sentences.append((current_sentence, current_sentence_start_time, current_sentence_end_time))

    # Step 2: Merge sentences into paragraphs
    paragraphs = []
    current_paragraph = []
    current_paragraph_start_time = None
    current_paragraph_end_time = None

    # Iterate over the identified sentences
    for sentence, start_time, end_time in sentences:
        # Calculate the duration of the current sentence (in seconds)
        sentence_duration = (end_time - start_time) / 1000.0

        # If no paragraph has been started, initialize the start and end times
        if current_paragraph_start_time is None:
            current_paragraph_start_time = start_time
            current_paragraph_end_time = end_time

        # Calculate the new duration if the current sentence is added to the paragraph
        new_duration = (end_time - current_paragraph_start_time) / 1000.0

        # If the new duration exceeds the maximum allowed duration, save the current paragraph
        if new_duration > max_duration:
            paragraphs.append((" ".join(current_paragraph), current_paragraph_start_time / 1000.0, current_paragraph_end_time / 1000.0))

            # Start a new paragraph with the current sentence
            current_paragraph = sentence
            current_paragraph_start_time = start_time
            current_paragraph_end_time = end_time
        else:
            # Merge the current sentence into the paragraph
            current_paragraph.extend(sentence)
            current_paragraph_end_time = end_time

    # Add the last paragraph (if present)
    if current_paragraph:
        paragraphs.append((" ".join(current_paragraph), current_paragraph_start_time / 1000.0, current_paragraph_end_time / 1000.0))

    return paragraphs

def split_and_add_sentences(current_sentence, start_time, end_time, sentences, max_duration):
    """Helper function to split a long sentence into smaller parts based on max_duration."""
    current_length = 0
    new_sentence = []
    sentence_start_time = start_time

    for word in current_sentence:
        word_start_time = sentence_start_time
        word_end_time = word_start_time + (end_time - start_time) / len(current_sentence)

        new_sentence.append(word)
        current_length += (word_end_time - word_start_time) / 1000.0

        if current_length >= max_duration:
            sentences.append((new_sentence, sentence_start_time, word_end_time))
            new_sentence = []
            sentence_start_time = word_end_time

    if new_sentence:
        sentences.append((new_sentence, sentence_start_time, end_time))

# # Load corresponding text annotations from TextGrid
# tgrid = tg.TextGrid.fromFile('/nas/nas_32/AI-being/zhangjz/exp_motion/datasets/beat2_original/raw_data/beat_v2.0.0/beat_english_v2.0.0/textgrid/1_wayne_0_83_83.TextGrid')
#
# # Call the function using your tgrid
# paragraphs = split_and_merge_sentences(tgrid[0].intervals, max_duration=10.0)
#
# # Print the generated paragraphs with time intervals in seconds
# for i, (paragraph, start, end) in enumerate(paragraphs):
#     print(f"Paragraph {i+1}: [{start:.2f} - {end:.2f} seconds] {paragraph}")