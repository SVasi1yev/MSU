def squares(w, h, *sqrts):
    field = []
    for i in range(h):
        field.append(['.'] * w)
    for sqrt in sqrts:
        # print(sqrt)
        for i in range(sqrt[1], sqrt[1] + sqrt[2]):
            for j in range(sqrt[0], sqrt[0] + sqrt[2]):
                field[i][j] = sqrt[3]
                
    for i in range(len(field)):
        print(''.join(field[i]))
    
# squares(20,23,(1,1,5,'@'), (2,3,1,'!'), (4,5,11,'#'), (8,11,9,'/'))