# Get all the idcs of dogs, cats and snakes, tench in the imagenet dataset

from data.imagenet_classnames import name_map

lists_to_check = ['dog', 'cat', 'snake', 'tench', 'lion', 'leopard', 'cheetah', 'zebra', 'hippo']

list_to_avoid = ['fence', 'dragonfly', 'sea', 'oyster', 'Cacatua', 'hot', 'CRO', 'catamaran', 'frog', 'sled', 'monkey']
def get_idcs():

    idcs = []
    for i in range(1000):
        name = name_map[i]

        if len([x for x in list_to_avoid if x in name]) > 0:
            continue

        if len([x for x in lists_to_check if x in name]) > 0:
            idcs.append(i)
            #print(i, name_map[i])
    print(idcs)
    #Repeat the idcs 5 times, with bases from 1000 to 5000
    idcs_bases = []
    for i in range(5):
        idcs_bases.append([x + 1000 * (i+1) for x in idcs])

    [idcs.extend(x) for x in idcs_bases]

    return idcs


if __name__ == '__main__':
    idcs = get_idcs()
    print(len(idcs))
    # print idces in this format 
    # "- idx1 \n - idx2 \n - idx3 \n ..."
    print("- " + "\n - ".join([str(x) for x in idcs]))