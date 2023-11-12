with open("final.csv", 'w') as w:
    with open("result.csv", 'r') as f:
        for i in f.read().replace('"', '').split('\n'):
            a = i.split(',')
            print(a)
            w.write(f'"{a[0]}", "{a[1]}", "{a[2]}", "{a[3]}"\n')