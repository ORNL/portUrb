
nodes = [1,2,4,8,16,32]
clocks = [501,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700]
for node in nodes :
  for clock in clocks :
    kwhr = 0.
    for i in range(node) :
      fh1 = open(f"nodes_{node:02d}/maxsclk_{clock}/energy_start_node_{i}.txt","r")
      fh2 = open(f"nodes_{node:02d}/maxsclk_{clock}/energy_end_node_{i}.txt"  ,"r")
      j1 = int(fh1.readline().split()[0])
      j2 = int(fh2.readline().split()[0])
      kwhr += (j2-j1)/3.6e6
    fh3 = open(f"nodes_{node:02d}/maxsclk_{clock}/app_output.txt"  ,"r")
    for line in fh3 :
      if (line[:4] == "main") :
        time = float(line.split()[2])
    # print(f"Nodes: {node:02d} , Clock: {clock:04d} , KWhr: {kwhr:.5f} , Time: {time:.5e}")
    print(f"{node:02d} {clock:04d} {kwhr:.5f} {time:.5e}")

