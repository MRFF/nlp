import requests
import re
import pickle
import copy


base_url = 'https://baike.baidu.com'


def match_in_url(url, pattern_str):
	"""
	url: url to get page source from
    pattern_str: regular expression string to be matched
	return: matched result
	"""
    response = requests.get(url, headers='')
    response.encoding = 'utf-8'
    pattern = re.compile(pattern_str)
    result = re.findall(pattern, response.text)
    return result

# 从百度百科获得所有地铁线路的url
def get_line_hrefs():
    line_url = match_in_url(base_url + '/item/%E5%8C%97%E4%BA%AC%E5%9C%B0%E9%93%81/408485',
                '<a target=_blank href="(/item/.+?)">(北京地铁.+?线)</a>') # 在"*","?","+","{m,n}"后面加上？，使贪婪变成非贪婪
    line_hrefs = {}
    line_stations = {}
    for href, name in line_url:
        line_hrefs[name] = base_url + href

    return line_hrefs

# 获得每条线路的站点
def get_line_stations():
    line_urls = get_line_hrefs()
    lines = {}
    lines['1'] = match_in_url(line_urls['北京地铁1号线'],'<td align="center" valign="middle" colspan="1" rowspan="1">(\w+)</td>')
    lines['2'] = match_in_url(line_urls['北京地铁2号线'],'<tr><th>(\w+?)</th><td width="72" align="middle" valign="center">[0-9]{2}:[0-9]{2}</td>')
    lines['4'] = match_in_url(line_urls['北京地铁4号线'], '<td align="center" valign="middle" colspan="1" rowspan="1">(\w+?)</td><td align="center" valign="middle">.+?</td>') + ['公益西桥']
    lines['5'] = match_in_url(line_urls['北京地铁5号线'],'<th width="88" align="center" valign="middle">(\w+)</th>')
    lines['6'] = match_in_url(line_urls['北京地铁6号线'],'<th>(\w+)</th>')[3:]
    lines['7'] = match_in_url(line_urls['北京地铁7号线'],'<th>(\w+)</th>')[2:]
    lines['8'] = match_in_url(line_urls['北京地铁8号线'],'<th width="[0-9]+">(\w+)</th>')[5:]
    lines['9'] = match_in_url(line_urls['北京地铁9号线'],'<th>(\w+)</th>')
    lines['10'] = match_in_url(line_urls['北京地铁10号线'],'<td width="84" align="center" valign="middle"><a target=_blank href=.+?>(\w+?)</a></td>')
    lines['13'] = match_in_url(line_urls['北京地铁13号线'],'<th align="center" valign="middle">(\w+)</th>')
    lines['14'] = match_in_url(line_urls['北京地铁14号线'],'<tr><th align="center" valign="middle">(\w+)</th><td width="[0-9]+" align="center" valign="middle">.*?</td>')
    print(lines['14'])
    lines['15'] = match_in_url(line_urls['北京地铁15号线'],'<th align="center" valign="middle">(\w+?)</th>')[4:]
    lines['16'] = match_in_url(line_urls['北京地铁16号线'],'<th>(\w+?)</th>')[5:]
    lines['八通'] = match_in_url(line_urls['北京地铁八通线'],'<th align="center" valign="middle">(\w+?)</th>')[4:]
    lines['昌平'] = match_in_url(line_urls['北京地铁昌平线'],'<td width="86" align="center" valign="middle">(\w+?)</td>')
    lines['昌平'].pop(1) 
    lines['大兴'] = match_in_url(line_urls['北京地铁大兴线'],'<td width="88" align="center" valign="middle">(\w+?)</td>')
    lines['房山'] = match_in_url(line_urls['北京地铁房山线'],'<th align="center" valign="middle">(\w+?)</th>')[3:] + ['阎村东']
    lines['亦庄'] = [stat[:-1] if stat != '亦庄火车站' else stat for stat in match_in_url(line_urls['北京地铁亦庄线'],'><a target=_blank href="/item/.+?" data-lemmaid="[0-9]+?">(\w+?)</a>') if stat.endswith('站') and len(stat) >= 3 ][2:]  
    lines['机场'] = [stat[:-1] if '东直门' in stat else stat for stat in match_in_url(line_urls['北京地铁机场线'],'<th align="center" valign="middle">(\w+?)</th>')[2:]]
    return lines

def find_wanted_path(start, dest, connection_graph):
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
        # paths = condition(paths) # 加上控制条件

# 将线路-车站信息写入文件，方便下次读取
# lines = get_line_stations()
# with open('line_stations', 'wb') as f:
#     pickle.dump(lines, f)

get_line_stations()
with open('line_stations', 'rb') as f:
    line_stations = pickle.load(f)

# 给每一站创建整个线路的对应列表
stat_stations = dict()
for stations in line_stations.values():
    for s in stations:
        # 线路中有换乘站，会出现在多条线路中。如果已经更新过，则遍历到后不再更新，否则后一次会覆盖前一次
        if stat_stations.get(s): continue
        stat_stations[s] = stations

# 创建站点-前后站的对应关系    
stat_connection = dict()
for stat in stat_stations.keys():
    stat_connection[stat] = set()

for stat in stat_stations.keys():
    # 换乘站会出现在多条线路中，需要将其每条线路中的前后站加入，所以要遍历每一条线路
    for stations in stat_stations.values():
        # 取索引时,如果站点不在当前线路中，会遇到ValueError，需要处理
        try:
            index = stations.index(stat)
            if index == 0:
                stat_connection[stat].add(stations[index + 1])
            elif index == len(stations) -1:
                stat_connection[stat].add(stations[index - 1])
            else:
                stat_connection[stat].add(stations[index + 1])
                stat_connection[stat].add(stations[index - 1])
        except ValueError:
            continue

    

for k, v in  stat_connection.items():
    print(k, v)
# for k,v in d.items():
    # print(k ,v)
    # for s in stats:
    #     # 使用深拷贝，这里留心python的赋值方式，是贴标签，而非创建新对象，因此会改一处，而动所有
    #     # d[s] = copy.deepcopy(stats)
    #     d[s] = stats 

print(find_wanted_path('金安桥', '2号航站楼',stat_connection))
