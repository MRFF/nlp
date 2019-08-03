import re
from collections import defaultdict

coor_source = """
    {name:'兰州', geoCoord:[103.73, 36.03]},
    {name:'嘉峪关', geoCoord:[98.17, 39.47]},
    {name:'西宁', geoCoord:[101.74, 36.56]},
    {name:'成都', geoCoord:[104.06, 30.67]},
    {name:'石家庄', geoCoord:[114.48, 38.03]},
    {name:'拉萨', geoCoord:[102.73, 25.04]},
    {name:'贵阳', geoCoord:[106.71, 26.57]},
    {name:'武汉', geoCoord:[114.31, 30.52]},
    {name:'郑州', geoCoord:[113.65, 34.76]},
    {name:'济南', geoCoord:[117, 36.65]},
    {name:'南京', geoCoord:[118.78, 32.04]},
    {name:'合肥', geoCoord:[117.27, 31.86]},
    {name:'杭州', geoCoord:[120.19, 30.26]},
    {name:'南昌', geoCoord:[115.89, 28.68]},
    {name:'福州', geoCoord:[119.3, 26.08]},
    {name:'广州', geoCoord:[113.23, 23.16]},
    {name:'长沙', geoCoord:[113, 28.21]},
    //{name:'海口', geoCoord:[110.35, 20.02]},
    {name:'沈阳', geoCoord:[123.38, 41.8]},
    {name:'长春', geoCoord:[125.35, 43.88]},
    {name:'哈尔滨', geoCoord:[126.63, 45.75]},
    {name:'太原', geoCoord:[112.53, 37.87]},
    {name:'西安', geoCoord:[108.95, 34.27]},
    //{name:'台湾', geoCoord:[121.30, 25.03]},
    {name:'北京', geoCoord:[116.46, 39.92]},
    {name:'上海', geoCoord:[121.48, 31.22]},
    {name:'重庆', geoCoord:[106.54, 29.59]},
    {name:'天津', geoCoord:[117.2, 39.13]},
    {name:'呼和浩特', geoCoord:[111.65, 40.82]},
    {name:'南宁', geoCoord:[108.33, 22.84]},
    //{name:'西藏', geoCoord:[91.11, 29.97]},
    {name:'银川', geoCoord:[106.27, 38.47]},
    {name:'乌鲁木齐', geoCoord:[87.68, 43.77]},
    {name:'香港', geoCoord:[114.17, 22.28]},
    {name:'澳门', geoCoord:[113.54, 22.19]}
"""


# 用正则从数据源中提取数据
def get_location_info():
    city_location = {}

    pattern = re.compile("name:'(\w+)', geoCoord:\[(\d+.?\d+),\s(\d+.?\d+)\]")
    for line in coor_source.split('\n'):
        if line:
            city, long, lat = re.findall(pattern, line)[0]
            city_location[city] = (float(long), float(lat))
    return city_location


# 计算已知经纬度的两点之间的举例
def get_geo_distance(start, end):
    import math

    lat1, lon1 = start
    lat2, lon2 = end
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c
    return d


# 搜索起点到终点的路径
def find_path(start, dest, connection_graph):
    # 起初只有1条路，就是起始城市本身
    paths = [[start], ]
    visited = set() # 标记已探索过的地点
    while paths: # 还有路可走
        path = paths.pop(0) # 确定路线起点
        frontier = path[-1] # 确定路线终点
        successors = connection_graph[frontier] # 得出其连接点
        for city in successors:
            if city in path: continue # 连接点已探索过，则跳出此循环，换连接点里的下一个
            new_path = path + [city]  # 路线中添加该连接点
            paths.append(new_path)    # 有了新路线，加入路线中，为下次探索准备
            if city == dest:
                return new_path # 如果该连接点就是终点，则返回这条线路，就是结果
            
        visited.add(frontier) # 探索完成，路线终点加入已探索路线
        # print(paths)

# 广度优先搜搜
def bfs(graph, start):
    # for print use
    flag = 0
    
    to_be_seen = [start]
    seen = set()
    while to_be_seen:
        flag += 1
        
        frontier = to_be_seen.pop(-1) # 先弹出哪个也决定顺序，
        if frontier in seen: continue
        for node in graph[frontier]:
            if node in seen: continue
            print(flag * '- ', node)
            to_be_seen = [node] + to_be_seen # 新发现的点放在最前，晚点再扩展
        seen.add(frontier)    
    return seen   


# 深度优先搜索
def dfs(graph, start):
    # for print use
    flag = 0 
    
    to_be_seen = [start]
    seen = set()
    while to_be_seen:
        flag += 1
        
        frontier = to_be_seen.pop(-1) # 先弹出哪个也决定顺序，队列
        if frontier in seen: continue
        for node in graph[frontier]:
            if node in seen: continue
            print(flag * '- ', node)
            to_be_seen.append(node) # 新发现的节点放在最后，最先扩展
        seen.add(frontier)    
    return seen   

# 将距离小于threshold的地点连接，生成图 
def create_connection_graph(threshold=320):
    city_connection_graph = defaultdict(set)
    for city1, location1 in city_location.items():
        for city2, location2 in city_location.items():
            if city1 == city2: continue
            distance = get_geo_distance(location1, location2)
            if distance < threshold:
                city_connection_graph[city1].add(city2)
                city_connection_graph[city2].add(city1)
    return city_connection_graph


# 加排序函数，控制搜索
def find_wanted_path(start, dest, connection_graph, condition):
    paths = [[start], ]
    visited = set()
    while paths: 
        path = paths.pop(0) 
        frontier = path[-1] 
        successors = connection_graph[frontier] 
        for city in successors:
            if city in path: continue 
            new_path = path + [city]  
            paths.append(new_path)    
            if city == dest: return new_path 
            
        visited.add(frontier) 
        paths = condition(paths) # 加上控制条件


def transfer_least_first(paths):
    return sorted(paths, key=len)
def shortest_path_first(paths):
    def get_path_distance(path):
        distance = 0
        for num, city in enumerate(path[:-1]):
            distance += get_geo_distance(city, path[num + 1])
        return distance

    return sorted(paths, key=get_geo_distance)       
def transfer_most_first(paths):
    return sorted(paths, key=len, reverse=True)




city_location = get_location_info()
example_conn_info = {
    '北京':['太原', '沈阳'],
    '太原':['北京', '西安', '郑州'],
    '兰州':['西安'],
    '西安':['兰州','长沙'],
    '长沙':['福州', '南宁'],
    '沈阳':['北京'],
    '郑州':['太原']
}



# 使用defaultdict，避免键找不到报错
example_conn = defaultdict(list)
example_conn.update(example_conn_info)
conn_graph = create_connection_graph()
print(find_path('北京', '台湾', conn_graph))
print(find_wanted_path('北京', '台湾', conn_graph, transfer_most_first))
