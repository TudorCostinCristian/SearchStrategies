import json
import time
import os, psutil
from copy import copy, deepcopy
from heapq import heappop, heappush

#---------------------------------------------------------------- FUNCTII NECESARE

def ReadData(test): # Citire test
    file = open(test)
    data = json.load(file)
    return data

def WriteData(output):
    file = open("output.json", 'w')
    json.dump(output, file)
    print(output)

def CreateInitialMatrix(): # returneaza o matrice cu height linii si width coloane, fiecare element avand valoarea -1
    matrix = []
    for i in range(0, height):
        line = []
        for j in range(0, width):
            line.append(-1)
        matrix.append(line)      
    return matrix

def IsLineFull(line): # Verifica daca ultimul element al unei linii este 0 (pentru a-l putea muta inainte unui sir de 1)
    if line[-1] == 0:
        return False
    return True

def IsColumnFull(col): # similar, pentru o coloana
    if col[-1] == 0:
        return False
    return True

def AddZero(line, pos): # elimina ultimul element din line(se apeleaza doar daca ultimul element din line e 0) si adauga un 0 inaintea sirului de 1 cu indicele pos
    line.pop()
    cnt = 0
    index = 0
    last_val = 0
    for val in line:
        if cnt == pos:
            break
        if val == 0 and last_val == 1:
            cnt = cnt + 1
        last_val = val
        index = index + 1
    line.insert(index, 0)
    return line

def CheckSolution(mat): # returneaza True daca matricea mat respecta toate restrictiile 
                        # impuse atat pentru linii, cat si pentru coloane. Altfel, returneaza False.
    rs = []
    cs = []
    for line in mat:
        row = []
        cnt = 0
        for val in line:
            if val == 1:
                cnt = cnt + 1
            else:
                if cnt != 0:
                    row.append(cnt)
                    cnt = 0
        if cnt != 0:
            row.append(cnt)
        rs.append(row)
    if rs != rows:
        return False

    for column in range(0, width):
        col = []
        cnt = 0
        for line in mat:
            if line[column] == 1:
                cnt = cnt + 1
            else:
                if cnt != 0:
                    col.append(cnt)
                    cnt = 0
        if cnt != 0:
            col.append(cnt)
        cs.append(col)
    if cs != columns:
        return False
    return True

def CheckPartialSolution(mat, ln): # primeste o matrice completata pana la linia ln si returneaza True daca aceasta 
                                   # respecta(pana la linia ln, inclusiv) restrictiile impuse pentru coloane. Altfel, returneaza False;
    for column in range(0, width):
        col = []
        cnt = 0
        for i in range(0, ln + 1):
            if mat[i][column] == 1:
                cnt = cnt + 1
            else:
                if cnt != 0:
                    col.append(cnt)
                    cnt = 0
        if cnt != 0:
            col.append(cnt)
        length = len(col)

        if length > len(columns[column]):
            return False

        for i in range(0, length - 1):
            if col[i] != columns[column][i]:
                return False
        if length > 0:
            if col[length - 1] > columns[column][length - 1]:
                return False
    return True

def GetLeftMostPerm(num_vals, length): # Primeste o lista cu valori numerice reprezentand lungimile sirurilor de 1 pentru o linie/coloana si returneaza
                                       # o lista cu length elemente in care toate sirurile de 1 sunt grupate la stanga, cu un singur spatiu intre ele (permutarea cea mai din stanga)
    perm = []
    row_copy = deepcopy(num_vals)
    for i in range(0, length):
        if len(row_copy) == 0:
            perm.append(0)
        else:
            if row_copy[0] == 0:
                row_copy.pop(0)
                perm.append(0)
            else:
                row_copy[0] = row_copy[0] - 1
                perm.append(1)
    return perm

def GetLinePerms(ln): # Primeste indicele unei linii din matrice si genereaza, folosind BFS, toate permutarile 
                      # care respecta restrictiile impuse pentru acea linie, pornind de la permutarea cea mai din stanga. 
    init_line = GetLeftMostPerm(rows[ln], width)

    queue = []
    visited = []
    queue.append(init_line)
    visited.append(init_line)

    while len(queue) > 0:
        line = queue.pop(0)
        linePerms[ln].append(line)
        if IsLineFull(line) == False:
            for p in range(0, len(rows[ln])):
                new_perm = deepcopy(line)
                new_perm = AddZero(new_perm, p)
                if new_perm not in visited:
                    queue.append(new_perm)
                    visited.append(new_perm)
    return linePerms

def GetColumnPerms(cl): # similar, pentru o coloana
    init_column = GetLeftMostPerm(columns[cl], height)

    queue = []
    visited = []
    queue.append(init_column)
    visited.append(init_column)

    while len(queue) > 0:
        col = queue.pop(0)
        colPerms[cl].append(col)
        if IsColumnFull(col) == False:
            for p in range(0, len(columns[cl])):
                new_perm = deepcopy(col)
                new_perm = AddZero(new_perm, p)
                if new_perm not in visited:
                    queue.append(new_perm)
                    visited.append(new_perm)
    return colPerms

#---------------------------------------------------------------- BFS

def BFS():
    global matrix
    global nodes_generated
    global nodes_expanded
    queue = []
    queue.append((0, []))

    while len(queue) > 0:
        current = queue.pop(0)
        ln = current[0]
        perms = current[1]
        for i in range(0, ln):
            matrix[i] = linePerms[i][perms[i]]
        if ln < height:
            if CheckPartialSolution(matrix, ln - 1) == True:
                for i in range(0, len(linePerms[ln])):
                    new_perms = deepcopy(perms)
                    new_perms.append(i)
                    queue.append((ln + 1, new_perms))
                    nodes_generated = nodes_generated + 1
                
        if ln == height:
            if CheckSolution(matrix) == True:
                return True
        nodes_expanded = nodes_expanded + 1
    return

#---------------------------------------------------------------- DFS

def DFS(ln):     
    global nodes_generated
    global nodes_expanded
    if ln == height:
        if CheckSolution(matrix) == True:
            return True
        else: 
            return False
    if ln < height and ln > 0:
        if CheckPartialSolution(matrix, ln - 1) == False:
            return False

    for i in range(0, len(linePerms[ln])):
        matrix[ln] = linePerms[ln][i]
        nodes_generated = nodes_generated + 1
        if DFS(ln + 1) == True:
            return True
    nodes_expanded = nodes_expanded + 1
    return False

#---------------------------------------------------------------- ITERATIVE DEEPENING

def DFS_ID(ln, AdMax):
    global nodes_generated
    global nodes_expanded
    if ln == height:
        if CheckSolution(matrix) == True:
            return True
        else: 
            return False
    if ln < height and ln > 0:
        if CheckPartialSolution(matrix, ln - 1) == False:
            return False

    if ln == AdMax:
        return False

    for i in range(0, len(linePerms[ln])):
        matrix[ln] = linePerms[ln][i]
        nodes_generated = nodes_generated + 1
        if DFS_ID(ln + 1, AdMax) == True:
            return True
    nodes_expanded = nodes_expanded + 1
    return False

def IterativeDeepening():
    AdMax = 0
    while DFS_ID(0, AdMax) == False:
        AdMax = AdMax + 1
    return

#---------------------------------------------------------------- A*

def GetColPermsSum():
    m = []
    for i in range(0, height):
        m.append([])
        for j in range(0, width):
            m[i].append(0)
    for l in range(0, height):
        for c in range(0, width):
            for perm in colPerms[c]:
                m[l][c] = m[l][c] + perm[l]
    return m 

def Heur1_ColumnPermsSum(sol):
    ln = sol[0]
    perms = sol[1]
    if ln == 0:
        return 0
    cost = 0
    for i in range(0, width):
        if linePerms[ln - 1][perms[ln - 1]][i] == 1:
            cost = cost + len(colPerms[i]) - colPermsSum[ln - 1][i]
        else:
            cost = cost + colPermsSum[ln-1][i]
    return cost / ln

def Heur2_NeighborsAbove(sol):
    ln = sol[0]
    perms = sol[1]
    if ln == 0:
        return 0
    cost = 0
    prev_line = 0
    curr_line = 0
    if ln > 2:
        for i in range(0, width):
            prev_line = prev_line + linePerms[ln-2][perms[ln-2]][i]
            curr_line = curr_line + linePerms[ln-1][perms[ln-1]][i]
            if linePerms[ln-2][perms[ln-2]][i] == 1:
                cost = cost + 1 - linePerms[ln-1][perms[ln-1]][i]

    if prev_line != 0 and curr_line != 0:
        cost = cost / (prev_line/curr_line)

    return cost

def astar(heur):
    global nodes_generated
    global nodes_expanded
    frontier = []
    sol = (0, [])
    heappush(frontier, (heur(sol), sol))

    while frontier:
        current = heappop(frontier)
        cost_est = current[0]
        sol = current[1] 
        ln = sol[0]
        perms = sol[1]

        for i in range(0, ln):
            matrix[i] = linePerms[i][perms[i]]

        if ln == height:
            if CheckSolution(matrix) == True:
                return True

        if ln < height:
            if CheckPartialSolution(matrix, ln - 1) == True:
                for i in range(0, len(linePerms[ln])):
                    new_perms = deepcopy(perms)
                    new_perms.append(i)
                    new_sol = (ln + 1, new_perms)
                    nodes_generated = nodes_generated + 1
                    heappush(frontier, (heur(new_sol), new_sol))
        nodes_expanded = nodes_expanded + 1
    return

#---------------------------------------------------------------- MAC BKT

def MAC():
    queue = []
    for var in Vars:
        for constraint in get_constraints(var, Constraints):
            queue.append((var,constraint))

    while len(queue) > 0:
        el = queue.pop(0)
        var1 = el[0]
        constr = el[1]
        constr_vars = constr[0]
        for var in constr_vars:
            if var != var1:
                var2 = var
                break
        
        domain_changed = 0
        for val1 in Domains[var1]:
            flag = 0
            for val2 in Domains[var2]:
                if check_constraint({var1 : val1, var2 : val2}, constr) == True:
                    flag = 1
                    break
            if flag == 0:
                Domains[var1].remove(val1)
                domain_changed = 1
        if domain_changed == 1:
            for constraint in get_constraints(var1, Constraints):
                aux = constraint[0]
                for var in aux:
                    if var != var1:
                        var_aux = var
                        break
                if (var1, constraint) not in queue:
                    queue.append((var1, constraint))
                if (var_aux, constraint) not in queue:
                    queue.append((var_aux, constraint))          
    return

def get_constraints(var, constraints):
    res = []
    for c in constraints:
        if var in c[0]:
            res.append(c)
    return res

def fixed_constraints(solution, constraints):
    res = []
    for c in constraints:
        flag = 0
        for var in c[0]:
            if var not in solution.keys():
                flag = 1
                break
        if flag == 0:
            res.append(c)
    return res

def check_constraint(solution, constraint):
    args = []
    for var in constraint[0]:
        args.append(solution[var])
    return constraint[1](*args)

def BKT(vars, domains, constraints, solution):
    global final_solution
    global nodes_generated
    global nodes_expanded
    nodes_generated = nodes_generated + 1
    if not vars:
        final_solution = solution
        return True
    elif not domains[vars[0]]:
        return False
    else:
        nodes_expanded = nodes_expanded + 1
        var = vars[0]
        val = domains[var].pop(0)

        new_solution = copy(solution)
        new_solution[var] = val

        constrs = fixed_constraints(new_solution, constraints)
        flag = 0
        for c in constrs:
            if check_constraint(new_solution, c) == False:
                flag = 1

        if flag == 0:
            new_vars = deepcopy(vars)
            new_vars.pop(0)
            if BKT(new_vars, deepcopy(domains), constraints, new_solution) == True:
                return True

        return BKT(vars, deepcopy(domains), constraints, solution)

#---------------------------------------------------------------- REZOLVARE

def SolveProblem(method, test):
    global width
    global height
    global rows
    global columns
    global linePerms
    global colPerms
    global matrix

    global nodes_generated
    global nodes_expanded

    nodes_generated = 0
    nodes_expanded = 0

    data = ReadData(test)
    width = data['width']
    height = data['height']
    rows = data['rows']
    columns = data['columns']

    linePerms = []
    colPerms = []

    for i in range(0, height):
        linePerms.append([])
        GetLinePerms(i)
    for i in range(0, width):
        colPerms.append([])
        GetColumnPerms(i)

    matrix = CreateInitialMatrix()

    start_time = time.time()

    if method == "DFS":
        DFS(0)
    elif method == "BFS":
        BFS()
    elif method == "ID":
        IterativeDeepening()
    elif method == "ASTAR_HEUR1":
        global colPermsSum
        colPermsSum = GetColPermsSum()
        astar(Heur1_ColumnPermsSum)
    elif method == "ASTAR_HEUR2":
        astar(Heur2_NeighborsAbove)
    elif method == "MAC_BKT":
        global Vars
        global Domains
        global Constraints
        Vars = []
        Domains = {}
        Constraints = []
        for i in range(0, height):
            Vars.append("L" + str(i))
            Domains["L" + str(i)] = linePerms[i]

        for i in range(0, width):
            Vars.append("C" + str(i))
            Domains["C" + str(i)] = colPerms[i]

        for l in range(0, height):
            for c in range(0, width):
                row_str = "L" + str(l)
                col_str = "C" + str(c)
                Constraints.append(([row_str, col_str], lambda row, col, c=c, l=l: row[c] == col[l]))

        MAC()
        global final_solution
        final_solution = {}
        if BKT(Vars, deepcopy(Domains), Constraints, {}):
            for i in range(0, height):
                matrix[i] = final_solution["L" + str(i)]

    total_time = time.time() - start_time
    time_str = str(total_time) + " seconds"

    return (matrix, time_str, method, nodes_expanded, nodes_generated)

#--------------------------------------------------------------------------------------------------------------------------------------

tests = ["teste/test1.json", "teste/test2.json", "teste/test3.json", "teste/test4.json", "teste/test5.json", "teste/test6.json"]
methods = ["DFS", "BFS", "ID", "ASTAR_HEUR1", "ASTAR_HEUR2", "MAC_BKT"]

res = SolveProblem(methods[0], tests[0])

output = {"strategy": res[2], 
"nodes_generated": res[4],
"nodes_expanded": res[3],
"time": res[1],
"solution": res[0]}

WriteData(output)



