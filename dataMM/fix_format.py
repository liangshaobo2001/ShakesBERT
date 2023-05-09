

shakespeare_lines = []
punctuation = {"’", ",", "(", ")", ".", "”", "“", ";", "!", "?", "'"}
with open("Sonnets.txt", encoding='utf-8') as file:
    capture = True
    for line in file:
        if capture and not line.isspace():
            processed = line.strip()
            for punct in punctuation:
                if punct in processed:
                    split = processed.split(punct)
                    to_join = " " + punct
                    processed = to_join.join(split)
            processed = processed.lower()
            processed = processed.replace("’", "'")
            if processed[-1] in punctuation:
                processed = processed[:-1]
            shakespeare_lines.append(processed)
            # print(processed)
        if line.isspace():
            capture = True
        else:
            capture = False
            # print(processed)

num_removed = 0
with open("pretrain.txt", encoding="utf8") as f:
    post_processed = open("pretrain_edited.txt", "a", encoding='utf-8')
    # for line in f:
    #     print(line)
    for line in f:
        found = False
        for verse in shakespeare_lines:
            if verse in line.lower():
                found = True
                num_removed += 1
                print(verse)
                break
        if not found:
            post_processed.write(line)
    post_processed.write(str(num_removed))
    post_processed.close()

    
