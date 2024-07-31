






















    """
    context.append(question)
    input_text = " ".join(context)
    result = nlp(input_text)
    label = max(result[0], key=lambda x: x['score'])['label']
    return id2label[int(label.split('_')[-1])]