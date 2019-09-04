
# entities list for umls, YAGO3-10, FB15k-237, WN18RR, WN11, FB13
with open('data/WN18RR/train.tsv', 'r') as f, open('data/WN18RR/test.tsv', 'r') as f1, open('data/WN18RR/dev.tsv', 'r') as f2, open('data/WN18RR/entities.txt', 'w') as f3:
    lines = f.readlines() + f1.readlines() + f2.readlines()
    entities = set()
    for line in lines:
        line = line.strip()
        temp = line.split('\t')
        entities.add(temp[0])
        entities.add(temp[2])
    entities_str = '\n'.join(list(entities))
    f3.write(entities_str)

# relations list for FB15k-237, umls, WN18RR and YAGO3-10, WN11, FB13

with open('data/WN18RR/train.tsv', 'r') as f, open('data/WN18RR/test.tsv', 'r') as f1, open('data/WN18RR/dev.tsv', 'r') as f2, open('data/WN18RR/relations.txt', 'w') as f3:
    relations = set()
    lines = f.readlines() + f1.readlines() + f2.readlines()
    for line in lines:
        line = line.strip()
        temp = line.split('\t')
        relations.add(temp[1])
    relations_str = '\n'.join(relations)
    f3.write(relations_str)

# entities and relation list for WN18 and FB15K
with open('data/FB15K/entity2id.txt', 'r') as f, open('data/FB15K/relation2id.txt', 'r') as f1:
    lines = f.readlines()
    ent_id_dict = {}
    entities = []
    for line in lines:
        line = line.strip()
        temp = line.split()
        if len(temp) == 2:
            ent_id_dict[temp[1]] = temp[0]
            entities.append(temp[0])

    lines = f1.readlines()
    rel_id_dict = {}
    relations = []   
    for line in lines:
        line = line.strip()
        temp = line.split()
        if len(temp) == 2:
            rel_id_dict[temp[1]] = temp[0]
            relations.append(temp[0])

with open('data/FB15K/entities.txt', 'w') as f, open('data/FB15K/relations.txt', 'w') as f1:
    f.write('\n'.join(entities))
    f1.write('\n'.join(relations))

# train.tsv, dev.tsv and test.tsv for WN18 and FB15K
with open('data/FB15K/valid2id.txt', 'r') as f, open('data/FB15K/dev.tsv', 'w') as f1:
    lines = f.readlines()
    text_lines = []
    for line in lines:
        line = line.strip()
        temp = line.split()
        
        if len(temp) == 3:
            head_entity = ent_id_dict[temp[0]]
            tail_entity = ent_id_dict[temp[1]]
            relation = rel_id_dict[temp[2]]
            text_lines.append(head_entity + '\t' + relation + '\t'+ tail_entity)
    text_lines_str = '\n'.join(text_lines)
    f1.write(text_lines_str)

# entity to text for WN18RR
with open('data/WN18RR/wordnet-mlj12-definitions.txt', 'r') as f, open('data/WN18RR/entity2text.txt', 'w') as f1:
    lines = f.readlines()
    ent2texts = []
    count = 0
    
    for line in lines:
        line = line.strip()
        temp = line.split('\t')
        if len(temp) == 3:
            if temp[1].find('_NN_') != -1 or temp[1].find('_JJ_') != -1 or temp[1].find('_VB_') != -1 or temp[1].find('_RB_') != -1:
                wordnet_str = temp[1][2:]
                num_start = wordnet_str.rfind('_')
                pos_start = wordnet_str[:num_start].rfind('_')
                name = wordnet_str[:pos_start]
                name = name.replace('_', ' ')
                ent2texts.append(temp[0] + '\t' + name + ', ' + temp[2])
                count += 1
    print(count)
    f1.write('\n'.join(ent2texts))

# entity to text for FB, names
with open('data/FB15k-237/FB15k_mid2name.txt', 'r') as f, open('data/FB15k-237/entity2text.txt', 'w') as f1:
    lines = f.readlines()
    ent2texts = []

    for line in lines:
        line = line.strip()
        temp = line.split('\t')
        temp[1] = temp[1].replace('_', ' ')
        ent2texts.append(temp[0] + '\t' + temp[1])
    f1.write('\n'.join(ent2texts))

# entity to text for UMLS and YAGO3-10
with open('data/YAGO3-10/entities.txt', 'r') as f, open('data/YAGO3-10/entity2text.txt', 'w') as f1:
    lines = f.readlines()
    ent2texts = []

    for line in lines:
        line = line.strip()
        ent2texts.append(line + '\t' + line.replace('_', ' '))
    f1.write('\n'.join(ent2texts))

# relations to texts for Freebase, umls, WordNet
with open('data/WN18/relations.txt', 'r') as f, open('data/WN18/relation2text.txt', 'w') as f1:
    lines = f.readlines()
    relation_texts = []
    for line in lines:
        line = line.strip()
        text = line.replace('/' , ' ')
        text = text.replace('_', ' ').strip()
        #print(line, text)
        relation_texts.append(line + '\t' + text)
    f1.write('\n'.join(relation_texts))


# entity to text for FB, descriptions
with open('data/FB15K/FB15k_mid2description.txt', 'r') as f, open('data/FB15K/entity2textlong.txt', 'w') as f1:
    lines = f.readlines()
    ent2texts = []

    for line in lines:
        line = line.strip()
        temp = line.split('\t')
        temp[1] = temp[1][1:-4]
        temp[1] = temp[1].replace('_', ' ')
        ent2texts.append(temp[0] + '\t' + temp[1])
    f1.write('\n'.join(ent2texts))


# entity text length
max_len = 0
min_len = 10000
avg_len = 0
with open('data/WN18RR/entity2text.txt', 'r') as f1:
    lines = f1.readlines()
    for line in lines:
        temp = line.strip().split("\t")
        words = temp[1].split(" ")
        length = len(words)
        print(length)
        if length < min_len:
            min_len = length
        if length > max_len:
            max_len = length
        avg_len += length
    print("max:",  max_len)
    print("min:", min_len)
    print("avg:", avg_len * 1.0 / len(lines))
            