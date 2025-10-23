import os

from pymilvus import MilvusClient, DataType, Function, FunctionType
from openai import OpenAI
import json



# Author:@南哥AGI研习社 (B站 or YouTube 搜索“南哥AGI研习社”)


# 搜索和查询
# 1、大模型初始化
openai_client = OpenAI(
	base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
	api_key=os.getenv("DASHSCOPE_API_KEY")
)

# 2、实例化Milvus客户端对象
client = MilvusClient(
    uri="http://localhost:19530",
    db_name="jobs_rag"
)

# 3、定义文本embedding处理函数
def emb_text(text):
    return (
        openai_client.embeddings.create(input=text, model="text-embedding-v3")
        .data[0]
        .embedding
    )

# 4、ANN搜索
# 近似近邻（ANN）搜索以记录向量嵌入排序顺序的索引文件为基础
# 根据接收到的搜索请求中携带的查询向量查找向量嵌入子集，将查询向量与子群中的向量进行比较，并返回最相似的结果
question = "招聘java开发的岗位有哪些"
res = client.search(
    collection_name="job_postings",
    anns_field="embedding",
    data=[emb_text(question)],
    limit=2,
    search_params={"metric_type": "COSINE"},
    output_fields=["title", "job_name", "job_salary", "job_edu"]
)
retrieved_lines_with_distances = [
    (res["entity"]["title"], res["entity"]["job_name"], res["entity"]["job_salary"], res["entity"]["job_edu"], res["distance"]) for res in res[0]
]
print("问题："+question)
print("答案：")
print(json.dumps(retrieved_lines_with_distances, indent=4, ensure_ascii=False))
print()

# # 5、使用标准过滤再进行ANN搜索
# # 若集合中同时包含向量嵌入及其元数据，可以在 ANN 搜索之前过滤元数据，以提高搜索结果的相关性
# # 过滤符合搜索请求中过滤条件的实体，在过滤后的实体中进行 ANN 搜索
question = "学历要求是硕士及以上的岗位有哪些"
res = client.search(
    collection_name="job_postings",
    anns_field="embedding",
    data=[emb_text(question)],
    limit=3,
    filter='job_edu like "%硕士%"',
    search_params={"metric_type": "COSINE"},
    output_fields=["title", "job_name", "job_salary", "job_edu"]
)
retrieved_lines_with_distances = [
    (res["entity"]["title"], res["entity"]["job_name"], res["entity"]["job_salary"], res["entity"]["job_edu"], res["distance"]) for res in res[0]
]
print("问题："+question)
print("答案：")
print(json.dumps(retrieved_lines_with_distances, indent=4, ensure_ascii=False))
print()

# # 6、使用迭代过滤再进行ANN搜索
# ANN Search 对单次查询可调用的实体数量有最大限制，因此仅使用基本 ANN Search 可能无法满足大规模检索的需求。
# 对于 topK 超过 16,384 的 ANN Search 请求，建议考虑使用 SearchIterator。本节将介绍如何使用 SearchIterator 以及相关注意事项。
# 由于我们测试的数据量达不到这个量级，所以下面的搜索示例仅作参考，知道有这么个东西就行，官方文档：https://milvus.io/docs/zh/with-iterators.md
question = "学历要求是硕士及以上的岗位有哪些"
res = client.search(
    collection_name="job_postings",
    anns_field="embedding",
    data=[emb_text(question)],
    limit=3,
    filter='job_edu like "%硕士%"',
    output_fields=["title", "job_name", "job_salary", "job_edu"],
    search_params={
		"metric_type": "COSINE",
        "hints": "iterative_filter"
    }
)
retrieved_lines_with_distances = [
    (res["entity"]["title"], res["entity"]["job_name"], res["entity"]["job_salary"], res["entity"]["job_edu"], res["distance"]) for res in res[0]
]
print("问题："+question)
print("答案：")
print(json.dumps(retrieved_lines_with_distances, indent=4, ensure_ascii=False))
print()


# # 7、范围搜索
# # 执行范围搜索请求时，Milvus 以 ANN 搜索结果中与查询向量最相似的向量为圆心，以搜索请求中指定的半径为外圈半径，以range_filter为内圈半径，画出两个同心圆
# # 所有相似度得分在这两个同心圆形成的环形区域内的向量都将被返回
# # 这里，range_filter可以设置为0，表示将返回指定相似度得分（半径）范围内的所有实体
# 关于范围查找，radius参数的含义跟metric_type严格相关，具体可见：https://lxblog.com/qianwen/share?shareId=c3678b26-3677-47b7-a73b-2b7f629e509c
# 如果metric_type="COSINE"，则radius参数表示相似度得分（半径），即相似度得分高于该值的向量将被返回
# 如果metric_type="L2"，则radius参数表示距离（半径），即距离低于该值的向量将被返回
question = "学历要求是硕士及以上且月薪在30K及以上的岗位有哪些"
res = client.search(
    collection_name="job_postings",
    anns_field="embedding",
    data=[emb_text(question)],
    limit=2,
    filter='job_edu like "%硕士%"',
    output_fields=["title", "job_name", "job_salary", "job_edu"],
    search_params={
		"metric_type": "COSINE",
        "params": {
            "radius": 0.4,
            "range_filter": 0.6
        }
    }
)
retrieved_lines_with_distances = [
    (res["entity"]["title"], res["entity"]["job_name"], res["entity"]["job_salary"], res["entity"]["job_edu"], res["distance"]) for res in res[0]
]
print("问题："+question)
print("答案：")
print(json.dumps(retrieved_lines_with_distances, indent=4, ensure_ascii=False))
print()

# # 8、分组搜索
# # 分组搜索允许 Milvus 根据指定字段的值对搜索结果进行分组，以便在更高层次上汇总数据
# # 根据提供的查询向量执行 ANN 搜索，找到与查询最相似的所有实体，按指定的group_by_field 对搜索结果进行分组
# # 根据limit参数的定义，返回每个组的顶部结果，并从每个组中选出最相似的实体
question = "学历要求是硕士及以上且月薪在35K及以上的岗位有哪些"
res = client.search(
    collection_name="job_postings",
    anns_field="embedding",
    data=[emb_text(question)],
    limit=3,
    filter='job_edu like "%硕士%"',
    group_by_field="job_salary",
    group_size=2,
    output_fields=["title", "job_name", "job_salary", "job_edu"],
    # search_params={
	# 	"metric_type": "COSINE",
    #     "params": {
    #         "radius": 0.4,
    #         "range_filter": 0.6
    #     }
    # }
)
retrieved_lines_with_distances = [
    (res["entity"]["title"], res["entity"]["job_name"], res["entity"]["job_salary"], res["entity"]["job_edu"], res["distance"]) for res in res[0]
]
print("问题："+question)
print("答案：")
print(json.dumps(retrieved_lines_with_distances, indent=4, ensure_ascii=False))
print()


# # 9、获取查找持有指定主键的实体
# 相当于rds的=值查询
res = client.get(
    collection_name="job_postings",
    ids=[461688022328459973, 461688022328459974, 461688022328459981],
    output_fields=["id", "title", "job_name", "job_salary", "job_edu"],
)
print("结果：")
print(f"res:{res}")
print()