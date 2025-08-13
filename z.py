with open("data/labeled/for_echo_chamber_detection/indo_vaccination_labeled_136t_rawtext.csv", "r", encoding="utf-8-sig") as f:
    content = f.read()

with open("data/labeled/for_echo_chamber_detection/indo_vaccination_labeled_136t1.csv", "w", encoding="utf-8") as f:
    f.write(content)
