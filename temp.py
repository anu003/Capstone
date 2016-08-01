def get_combos(data):
    i = 0
    sent_combo = []
    for review in data:
        l = len(review)
        sent_combo.append(combo[i:i+l])
        i += l
    return sent_combo
