
sonnet_list = []
shakespeare_lines = []
quotation_mark = "’"
punctuation = {",", "(", ")", ".", "”", "“", ";", "!", "?", "'", ":", "‘"}
processed_sh = open("processed_sh_sonnets.txt", "a", encoding='utf-8')
# difficult_poems = open("difficult-sh.txt", "a", encoding='utf-8')
with open("Sonnets.txt", encoding='utf-8') as file:
    capture = True
    for line in file:
        if capture and not line.isspace():
            processed = line.strip()
            if quotation_mark in processed:
                split = processed.split(quotation_mark)
                to_join  = " " + quotation_mark
                processed = to_join.join(split)
            
            for punct in punctuation:
                if punct in processed:
                    split = processed.split(punct)
                    to_join = " " + punct + " "
                    processed = to_join.join(split)
            processed = processed.lower()
            processed = processed.replace("’", "'")
            processed = processed.replace("‘", "'")
            shakespeare_lines.append(processed)
            # print(processed)
        if line.isspace():
            if capture:
                sonnet_list.append(shakespeare_lines)
                shakespeare_lines = []
            capture = False
        else:
            capture = True
            # print(processed)

    sonnet_list.append(shakespeare_lines)

i = 0
for sonnet in sonnet_list:
    # print(sonnet)
    joined = " <eos> ".join(sonnet)
    header_footer = "<eos> " + joined + " <eos>"
    if i == 0:  
        print(header_footer)
        i+= 1
    processed_sh.write(header_footer)
    
processed_sh.close()
print(len(sonnet_list))