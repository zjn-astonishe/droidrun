def form_input_workflow(input_text='Audio Recorder'):
    """
    Workflow involving text input and UI interaction (8 actions)

    Args:
        input_text: Text to input (default: 'Audio Recorder') (required) (default: Audio Recorder)

    Returns:
        list: A list of action dictionaries
    """
    actions = []

    # Step 1: Tap element at index 12: 'Search' (FrameLayout) at coordinates (539, 2207)
    actions.append({
        "action": "Tap"
})

    # Step 2: Tap element at index 28: 'Search apps, web and more' (EditText) at coordinates (618, 228)
    actions.append({
        "action": "Tap"
})

    # Step 3: Input text: 'Audio Recorder' (clear=True)
    actions.append({
        "action": "Type",
        "text": "f\"{input_text}\""
})

    # Step 4: Tap element at index 3: 'Audio Recorder' (TextView) at coordinates (177, 469)
    actions.append({
        "action": "Tap"
})

    # Step 5: Tap element at index 8: 'RECORD' (Button) at coordinates (539, 2192)
    actions.append({
        "action": "Tap"
})

    # Step 6: Tap element at index 8: 'STOP' (Button) at coordinates (539, 2192)
    actions.append({
        "action": "Tap"
})

    # Step 7: Tap element at index 9: 'FINISH' (Button) at coordinates (663, 2192)
    actions.append({
        "action": "Tap"
})

    # Step 8: Tap element at index 6: 'Save' (Button) at coordinates (820, 1310)
    actions.append({
        "action": "Tap"
})

    return actions