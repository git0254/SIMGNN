#此文件用于测试
import os
import pandas as pd
from atlas_make_graph import collect_nodes_from_log
from atlas_make_graph import collect_edges_from_log
from atlas_make_graph import prepare_graph_new

def merge_properties(src_dict, target_dict):
    for k, v in src_dict.items():
        if k not in target_dict:
            target_dict[k] = v

def collect_dot_paths(base_dir):
    result = []
    for file in os.listdir(base_dir):   # 遍历
        full_path = os.path.join(base_dir, file)
        if os.path.isfile(full_path) and file.endswith(".dot"):  # 如果文件是 .dot 格式
            result.append(full_path)  # 使用 append 添加文件路径到列表
    return result  # 返回 .dot 文件路径的列表

def process_dot_files():
    # 加载一个数据集
    base_path = "graph_test"
    graph_file = collect_dot_paths(base_path)

    all_dfs = []    #列表
    all_netobj2pro = {}    #网络字典
    all_subject2pro = {}  # 进程字典
    all_file2pro = {}  # 文件字典
    processed_data = []
    domain_name_set = {}
    ip_set = {}
    connection_set = {}
    session_set = {}
    web_object_set = {}
    # TODO 0、ATLAS 数据集
    # 处理每个 .dot 文件
    for dot_file in graph_file:
        print(f"正在处理文件: {dot_file}")

        # 读取节点数据
        netobj2pro, subject2pro, file2pro, domain_name_set, ip_set, connection_set, session_set, web_object_set = collect_nodes_from_log(dot_file)

        # 打印节点数据
        print("netobj2pro:", netobj2pro)
        print("subject2pro:", subject2pro)
        print("file2pro:", file2pro)
        print("正在处理边: ")
        # 调用 `collect_edges_from_log` 收集边
        df = collect_edges_from_log(dot_file,domain_name_set, ip_set, connection_set, session_set, web_object_set, subject2pro, file2pro)  # 将 dot 文件传入收集边的函数
        df.to_csv("output.csv", index=False)
        print("edges:",df)

        # 只取良性前90%训练
        num_rows = int(len(df) * 0.9)
        df = df.iloc[:num_rows]  # 正确缩进
        all_dfs.append(df)
        merge_properties(netobj2pro, all_netobj2pro)
        merge_properties(subject2pro, all_subject2pro)
        merge_properties(file2pro, all_file2pro)

    # 训练用的数据集
    #将多个 DataFrame 合并成一个大的 DataFrame且去重
    atlas_df = pd.concat(all_dfs, ignore_index=True)
    atlas_df = atlas_df.drop_duplicates()

    # 将处理好后的数据集保存到 ATLAS.txt
    atlas_df.to_csv("ATLAS.txt", sep='\t', index=False)

    # 将整个图和相关特征准备好
    features, edges, mapp, relations, G = prepare_graph_new(atlas_df, all_netobj2pro, all_subject2pro, all_file2pro)

    # 打印准备好的图的输出
    print("features:", features)
    print("edges:", edges)
    print("mapp:", mapp)
    print("relations:", relations)
    print("G:", G)

# 主函数入口
if __name__ == "__main__":
    process_dot_files()
