import pandas as pd
import json
import igraph as ig
from type_enum import ObjectType
import re


# =================处理特征成图=========================
def add_node_properties(nodes, node_id, properties):
    if node_id not in nodes:
        nodes[node_id] = []
    nodes[node_id].extend(properties)


def update_edge_index(edges, edge_index, index, relations, relations_index):
    for src_id, dst_id in edges:
        src = index[src_id]
        dst = index[dst_id]
        edge_index[0].append(src)
        edge_index[1].append(dst)

        relation = relations[(src_id, dst_id)]
        relations_index[(src, dst)] = relation

def extract_properties(node_id, row, action, netobj2pro, subject2pro, file2pro):
    if node_id in netobj2pro:
        return netobj2pro[node_id]
    elif node_id in file2pro:
        return file2pro[node_id]
    elif node_id in subject2pro:
        return subject2pro[node_id]
    else:
        return [row.get('exec', ''), action] + ([row.get('path')] if row.get('path') else [])



# 成图+捕捉特征语料+简化策略这里添加
def prepare_graph(df):
    G = ig.Graph(directed=True)
    nodes, labels, edges, relations = {}, {}, [], {}
    # dummies = {"SUBJECT_PROCESS": 0, "MemoryObject": 1, "FILE_OBJECT_BLOCK": 2,
    #            "NetFlowObject": 3, "PRINCIPAL_REMOTE": 4, 'PRINCIPAL_LOCAL': 5}

    for _, row in df.iterrows():
        action = row["action"]
        properties = [row['exec'], action] + ([row['path']] if row['path'] else [])

        actor_id = row["actorID"]
        add_node_properties(nodes, actor_id, properties)
        labels[actor_id] = ObjectType[row['actor_type']].value

        object_id = row["objectID"]
        add_node_properties(nodes, object_id, properties)
        labels[object_id] = ObjectType[row['object']].value

        edge = (actor_id, object_id)
        edges.append(edge)
        relations[edge] = action

        # 初始化igraph的图
        G.add_vertices(1)
        G.vs[len(G.vs)-1]['name'] = actor_id
        G.vs[len(G.vs)-1]['type'] = ObjectType[row['actor_type']].value
        G.vs[len(G.vs)-1]['properties'] = properties
        G.add_vertices(1)
        G.vs[len(G.vs)-1]['name'] = object_id
        G.vs[len(G.vs)-1]['type'] = ObjectType[row['object']].value
        G.vs[len(G.vs)-1]['properties'] = properties
        G.add_edges([(actor_id, object_id)])
        G.es[len(G.es)-1]['actions'] = action

    features, feat_labels, edge_index, index_map, relations_index = [], [], [[], []], {}, {}
    for node_id, props in nodes.items():
        features.append(props)
        feat_labels.append(labels[node_id])
        index_map[node_id] = len(features) - 1

    update_edge_index(edges, edge_index, index_map, relations, relations_index)

    return features, feat_labels, edge_index, list(index_map.keys()), relations_index, G


# 成图+捕捉特征语料+简化策略这里添加
def prepare_graph_new(df, all_netobj2pro, all_subject2pro, all_file2pro):
    G = ig.Graph(directed=True)
    nodes, edges, relations = {}, [], {}

    for _, row in df.iterrows():
        action = row["action"]

        actor_id = row["actorID"]
        properties = extract_properties(actor_id, row, row["action"], all_netobj2pro, all_subject2pro, all_file2pro)
        add_node_properties(nodes, actor_id, properties)

        object_id = row["objectID"]
        properties1 = extract_properties(object_id, row, row["action"], all_netobj2pro, all_subject2pro, all_file2pro)
        add_node_properties(nodes, object_id, properties1)

        edge = (actor_id, object_id)
        edges.append(edge)
        relations[edge] = action

        ## 构建图
        # 点不重复添加
        actor_idx = get_or_add_node(G, actor_id, ObjectType[row['actor_type']].value, properties)
        object_idx = get_or_add_node(G, object_id, ObjectType[row['object']].value, properties)
        # 边也不重复添加
        add_edge_if_new(G, actor_idx, object_idx, action)

    features, edge_index, index_map, relations_index = [], [[], []], {}, {}
    for node_id, props in nodes.items():
        features.append(props)
        index_map[node_id] = len(features) - 1

    update_edge_index(edges, edge_index, index_map, relations, relations_index)

    return features, edge_index, list(index_map.keys()), relations_index, G


def get_or_add_node(G, node_id, node_type, properties):
    """
    查找图中是否已有节点 node_id：
    - 如果有，返回该节点索引，并更新属性
    - 如果没有，添加新节点并返回其索引
    """
    try:
        v = G.vs.find(name=node_id)
        v['properties'] = properties  # 可选更新属性
        return v.index
    except ValueError:
        G.add_vertex(name=node_id, type=node_type, properties=properties)
        return len(G.vs) - 1

def add_edge_if_new(G, src, dst, action):
    """
    向图 G 添加一条从 src 到 dst 的边，附带 action 属性。
    - 若边已存在且包含该 action，不做任何处理。
    - 若边已存在但未包含该 action 再添加一条边
    - 若边不存在，则添加边并设置 action。
    """
    if G.are_connected(src, dst):
        eids = G.get_eids([(src, dst)], directed=True, error=False)
        for eid in eids:
            if G.es[eid]["actions"] == action:
                return  # 该 action 已存在，不重复添加
    G.add_edge(src, dst)
    G.es[-1]["actions"] = action

def add_attributes(d, p):
    f = open(p)
    # data = [json.loads(x) for x in f if "EVENT" in x]
    # for test
    data = [json.loads(x) for i, x in enumerate(f) if "EVENT" in x and i < 300]
    info = []
    for x in data:
        try:
            action = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['type']
        except:
            action = ''
        try:
            actor = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['subject']['com.bbn.tc.schema.avro.cdm18.UUID']
        except:
            actor = ''
        try:
            obj = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObject'][
                'com.bbn.tc.schema.avro.cdm18.UUID']
        except:
            obj = ''
        try:
            timestamp = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['timestampNanos']
        except:
            timestamp = ''
        try:
            cmd = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['properties']['map']['cmdLine']
        except:
            cmd = ''
        try:
            path = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObjectPath']['string']
        except:
            path = ''
        try:
            path2 = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObject2Path']['string']
        except:
            path2 = ''
        try:
            obj2 = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObject2'][
                'com.bbn.tc.schema.avro.cdm18.UUID']
            info.append({'actorID': actor, 'objectID': obj2, 'action': action, 'timestamp': timestamp, 'exec': cmd,
                         'path': path2})
        except:
            pass

        info.append(
            {'actorID': actor, 'objectID': obj, 'action': action, 'timestamp': timestamp, 'exec': cmd, 'path': path})

    rdf = pd.DataFrame.from_records(info).astype(str)
    d = d.astype(str)

    return d.merge(rdf, how='inner', on=['actorID', 'objectID', 'action', 'timestamp']).drop_duplicates()


def collect_nodes_from_log(paths):#dot文件的路径
    #创建字典
    netobj2pro = {}
    subject2pro = {}
    file2pro = {}
    domain_name_set = {}
    ip_set = {}
    connection_set = {}
    session_set = {}
    web_object_set = {}
    nodes = []

    # 读取整个文件
    with open(paths, 'r', encoding='utf-8') as f:
        content = f.read()

    # 按分号分隔，处理每个段落
    statements = content.split(';')

    # 正则表达式匹配节点定义
    node_pattern = re.compile(r'^\s*"?(.+?)"?\s*\[.*?type="?([^",\]]+)"?', re.IGNORECASE)

    for stmt in statements:
        if 'capacity=' in stmt:
            continue  # 跳过包含 capacity 字段的段落
        match = node_pattern.search(stmt)
        if match:
            node_name = match.group(1)
            node_typen = match.group(2)
            nodes.append((node_name, node_typen))
    for node_name,node_typen in nodes:#遍历所有的节点
        node_id = node_name #节点id赋值
        node_type = node_typen #赋值type属性
        #-- 网络流节点 --
        if node_type == 'domain_name':
            nodeproperty = node_id
            netobj2pro[node_id] = nodeproperty
            domain_name_set[node_id]=nodeproperty
        if node_type == 'IP_Address':
            nodeproperty = node_id
            netobj2pro[node_id] = nodeproperty
            ip_set[node_id]=nodeproperty
        if node_type == 'connection':
            nodeproperty = node_id
            netobj2pro[node_id] = nodeproperty
            connection_set[node_id]=nodeproperty
        if node_type == 'session':
            nodeproperty = node_id
            netobj2pro[node_id] = nodeproperty
            session_set[node_id]=nodeproperty
        if node_type == 'web_object':
            nodeproperty = node_id
            netobj2pro[node_id] = nodeproperty
            web_object_set[node_id] = nodeproperty
        # -- 进程节点 --
        elif node_type == 'process':
            nodeproperty = node_id
            subject2pro[node_id]=nodeproperty
        # -- 文件节点 --
        elif node_type == 'file':
            nodeproperty = node_id
            file2pro[node_id]=nodeproperty

    return netobj2pro, subject2pro, file2pro, domain_name_set, ip_set, connection_set, session_set, web_object_set



def collect_edges_from_log(paths,domain_name_set, ip_set, connection_set, session_set, web_object_set, subject2pro, file2pro) -> pd.DataFrame:
    """
    从 DOT-like 日志文件中提取含 capacity 的边，并识别 source/target 属于哪个节点集合。
    返回一个包含 source、target、type、timestamp、source_type、target_type 的 DataFrame。
    """
    # 预定义的节点集合

    edges = []

    with open(paths, "r", encoding="utf-8") as f:
        content = f.read()

    statements = content.split(";")

    edge_pattern = re.compile(
        r'"?([^"]+)"?\s*->\s*"?(.*?)"?\s*\['
        r'.*?capacity=.*?'
        r'type="?([^",\]]+)"?.*?'
        r'timestamp=(\d+)',
        re.IGNORECASE | re.DOTALL
    )

    for stmt in statements:
        if "capacity=" not in stmt:
            continue
        m = edge_pattern.search(stmt)
        if m:
            source, target, edge_type, ts = (x.strip() for x in m.groups())

            # 判断 source/target 所属集合
            if source in domain_name_set:
                source_type = "NETFLOW_OBJECT"
            elif source in ip_set:
                source_type = "NETFLOW_OBJECT"
            elif source in connection_set:
                source_type = "NETFLOW_OBJECT"
            elif source in session_set:
                source_type = "NETFLOW_OBJECT"
            elif source in web_object_set:
                source_type = "NetFlowObject"
            elif source in  subject2pro:
                source_type = "SUBJECT_PROCESS"
            elif source in  file2pro:
                source_type = "FILE_OBJECT_BLOCK"
            else:
                source_type = "PRINCIPAL_LOCAL"

            if target in domain_name_set:
                target_type = "NETFLOW_OBJECT"
            elif target in ip_set:
                target_type = "NETFLOW_OBJECT"
            elif target in connection_set:
                target_type = "NETFLOW_OBJECT"
            elif target in session_set:
                target_type = "NETFLOW_OBJECT"
            elif target in web_object_set:
                target_type = "NetFlowObject"
            elif target in subject2pro:
                target_type = "SUBJECT_PROCESS"
            elif target in file2pro:
                target_type = "FILE_OBJECT_BLOCK"
            else:
                target_type = "PRINCIPAL_LOCAL"

            edges.append((source, source_type, target, target_type, edge_type, int(ts)))

    return pd.DataFrame(edges, columns=["actorID", "actor_type", "objectID", "object", "action", "timestamp"])