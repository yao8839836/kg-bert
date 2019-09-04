import wikipedia

s = wikipedia.summary("eicosanoid")
print(s)
print("eicosanoid".title())

entity_titles = []
entity_descriptions = []

with open("data/FB13/entities.txt") as f, open("data/FB13/entity2text_capital.txt", 'w') as f1:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        title = line.title().replace("_", " ")
        print(title)
        entity_titles.append(line + '\t' + title)
    f1.write('\n'.join(entity_titles))


with open("data/FB13/entities.txt") as f, open("data/FB13/entity2text.txt", 'w') as f2:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        try:
            description = wikipedia.summary(line)
        except:
            description = line.replace("_", " ").title()
        
        description = description.replace("\n", " ").replace("\t", " ")        
        print(description)
        if description.strip() == "":
            description = line.title().replace("_", " ")
        entity_descriptions.append(line + '\t' + description)

    f2.write('\n'.join(entity_descriptions))

with open("data/umls/entities.txt") as f, open("data/umls/entity2textlong.txt", 'w') as f2:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        try:
            description = wikipedia.summary(line)
        except:
            description = line.replace("_", " ").title()
        
        description = description.replace("\n", " ").replace("\t", " ")        
        print(description)
        if description.strip() == "":
            description = line.title().replace("_", " ")
        entity_descriptions.append(line + '\t' + description)

    f2.write('\n'.join(entity_descriptions))