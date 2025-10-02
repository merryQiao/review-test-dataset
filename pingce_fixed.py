import json
import os
import re
import time
import asyncio
import aiohttp
import aiofiles
import logging
import base64
import csv
import openai
from datetime import datetime

# === Robust score parser injected ===
def _parse_score(text: str):
    """
    从评语文本里尽可能鲁棒地抽取 0-10 分的整数得分。
    兼容：'评分：8'、'评分：8分'、'评分：8/10'、'Score: 8/10'、'Score 8 out of 10' 等。
    返回 int 或 None。
    """
    if not isinstance(text, str) or not text.strip():
        return None
    patterns = [
        r'(?:评分|得分|Score)\s*[:：]?\s*(\d{1,2})\s*(?:[/／]\s*10)?\s*(?:分)?\b',
        r'(?:Score)\s*[:：]?\s*(\d{1,2})\s*(?:out of|/)\s*10\b',
        r'(?:评分|得分)\s*[:：]?[^\d]{0,10}\s*(\d{1,2})\b',
    ]
    import re
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                val = int(m.group(1))
                return max(0, min(10, val))
            except:
                pass
    return None
# === End robust score parser ===



#openai.proxy = "socks5://127.0.0.1:7897"
# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('evaluation.log2', mode='w', encoding='utf-8')
    ]
)

# OpenAI 答题智能体
class AnsweringAgent:
    def __init__(self, config):
        self.model = config['model']
        self.temperature = config.get('temperature', 1)
        self.max_tokens = config.get('max_tokens', 2000)
        openai.api_key = config['api_key']
        openai.api_base = config['base_url']

    async def answer_question(self, question, image=None):
        system_prompt = (
            "你是一个有丰富经验的考古学和历史学研究专家。"
            "回答简答题时，请根据专业知识给出完整、准确的回答，严禁使用选项字母或简略格式。"
        )

        messages = [{"role": "system", "content": system_prompt}]

        # 处理有图和无图两种情况
        if image:
            img_base64 = base64.b64encode(image).decode("utf-8")
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                ]
            })
        else:
            messages.append({"role": "user", "content": question})

        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return response.choices[0].message["content"]

# OpenAI 评分智能体
class JudgingAgent:
    def __init__(self, config):
        self.model = config['model']
        self.temperature = config.get('temperature', 1)
        self.max_tokens = config.get('max_tokens', 2000)
        openai.api_key = config['api_key']
        openai.api_base = config['base_url']
    async def judge_answer(self, question, given_answer, reference_answer):
        messages = [
            {
                "role": "system",
                "content":
                (
                    "你是一位中华古文明领域的严格评分专家。你的输出必须是纯文本，不能使用 JSON、Markdown 表格、代码块、mermaid 或多余装饰。"
                    "请依据下述权重给出 0–10 分的总评，最后一行必须严格写成：评分：X分，其中 X 为整数。\n"
                    "评分维度与权重：\n"
                    "1) 内容完整性（40%）：对照参考答案的关键实体与要点，计算加权命中率。\n"
                    "   实体类型权重：核心概念=3；关键人物=2；历史事件=2；文化要素=1；时空标记=0.5。\n"
                    "2) 论证逻辑性（30%）：叙述是否有清晰因果、对比、层次与论证链条。\n"
                    "3) 知识准确性（20%）：历史事实、术语使用是否准确无误。\n"
                    "4) 表达规范性（10%）：用词规范、表述精炼，无口水话与跑题。\n"
                    "输出规则：\n"
                    "A. 只输出中文纯文本，不要任何列表符号、表格、代码围栏或额外标记。\n"
                    "B. 结构固定为四段，顺序如下：\n"
                    "   要点命中：……（简要列出考生答案覆盖到的关键要点/实体，按重要性总结，不编号）\n"
                    "   失分点：……（未覆盖或错误之处，简明扼要）\n"
                    "   维度评分：内容完整性=..；逻辑性=..；准确性=..；表达=..（均为 0–10 的小数或整数）\n"
                    "   评分：X分（将上述加权后的总分四舍五入为 0–10 的整数，只写这一行，不加其他文字）\n"
                    "C. 严格避免输出参考答案的逐字大段复述；仅做对比性归纳。\n"
                    "D. 如果考生答案严重跑题或为“未知/无”，则四维均可给低分，但仍需按模板输出，并保证“评分：X分”这一行存在。\n"
                )
            },
            {
                "role": "user",
                "content":
                (
                    f"问题：{question}\n"
                    f"考生答案：{given_answer}\n"
                    f"参考答案（用于对齐关键要点与实体，不要逐字照抄）：{reference_answer}"
                )
            }
        ]

        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message['content']


# 生成带时间戳的CSV文件名
def get_csv_filename():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"evaluation_results_{timestamp}.csv"

# 初始化CSV文件并写入表头
async def init_csv_file(filename):
    try:
        async with aiofiles.open(filename, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = [
                "question_id", "question", "subject_area",
                "model_answer", "expected_answer",
                "answer_type", "result", "score", "max_score",
                "assessment", "error", "processing_time", "timestamp"
            ]
            await f.write(",".join(header) + "\n")
    except Exception as e:
        logging.error(f"初始化CSV文件时出错: {str(e)}")
        return False
    return True

# 追加结果到CSV
async def append_to_csv(filename, result_data):
    try:
        async with aiofiles.open(filename, 'a', encoding='utf-8') as f:
            # 清理数据中的特殊字符
            cleaned_data = []
            for item in result_data:
                if isinstance(item, str):
                    # 移除换行符和多余的逗号
                    cleaned = item.replace('\n', ' ').replace('\r', ' ').replace('"', "'")
                    # 处理包含逗号的情况
                    if ',' in cleaned:
                        cleaned = f'"{cleaned}"'
                    cleaned_data.append(cleaned)
                else:
                    cleaned_data.append(str(item) if item is not None else "")

            await f.write(",".join(cleaned_data) + "\n")
    except Exception as e:
        logging.error(f"写入CSV时出错: {str(e)}")

# 加载配置文件
async def load_config1():
    try:
        async with aiofiles.open('openai_config.json', 'r') as f:
            contents = await f.read()
            return json.loads(contents)
    except FileNotFoundError:
        logging.error("未找到 openai_config.json 文件")
        exit(1)
    except json.JSONDecodeError:
        logging.error("openai_config.json 格式不正确")
        exit(1)
    except Exception as e:
        logging.error(f"加载配置时出错: {str(e)}")
        exit(1)
async def load_config2():
    try:
        async with aiofiles.open('judge_config.json', 'r') as f:
            contents = await f.read()
            return json.loads(contents)
    except FileNotFoundError:
        logging.error("未找到 judge_config.json 文件")
        exit(1)
    except json.JSONDecodeError:
        logging.error("judge_config.json 格式不正确")
        exit(1)
    except Exception as e:
        logging.error(f"加载配置时出错: {str(e)}")
        exit(1)

# 读取 QA 数据
async def load_qa_data(filename):
    qa_data = []
    try:
        async with aiofiles.open(filename, 'r', encoding='utf-8') as f:
            line_count = 0
            async for line in f:
                line_count += 1
                try:
                    data = json.loads(line)
                    # 添加唯一ID
                    data['id'] = f"{os.path.basename(filename)}_{line_count}"

                    # 确保学科领域字段存在
                    if "学科领域" not in data:
                        data["学科领域"] = "未分类"

                    qa_data.append(data)
                except json.JSONDecodeError:
                    logging.warning(f"跳过无法解析的行: {line.strip()}")
        logging.info(f"成功加载 {len(qa_data)} 个QA数据")
        return qa_data
    except FileNotFoundError:
        logging.error(f"未找到文件: {filename}")
        exit(1)
    except Exception as e:
        logging.error(f"加载QA数据时出错: {str(e)}")
        exit(1)

# 导出总结结果
async def export_summary(summary, filename="evaluation_summary2.txt"):
    try:
        async with aiofiles.open(filename, 'w', encoding='utf-8') as f:
            await f.write(summary)
    except Exception as e:
        logging.error(f"导出总结时出错: {str(e)}")

# 比较答案
def compare_answers(model_answer, expected_answer):
    try:
        # 清理和标准化答案
        def clean_answer(ans):
            return {item.strip().lower() for item in re.split(r'[,，]', ans.replace("\n", ",")) if item.strip()}

        model_set = clean_answer(model_answer)
        expected_set = clean_answer(expected_answer)

        return model_set == expected_set
    except Exception as e:
        logging.error(f"比较答案时出错: {str(e)}")
        return False

# 评估单个问题
async def evaluate_single_question(qa, agent, judging_agent, timeout=500):
    question = qa["Question"]
    expected_answer = qa["Final answer"]
    answer_type = qa.get("Answer Type", "exactMatch")
    question_id = qa.get("id", "unknown")
    subject_area = qa.get("学科领域", "未分类")  # 获取学科领域
    historical_period = qa.get("历史分期", "未知分期")
    core_purpose = qa.get("核心目的", "未知目的")
    image_path = qa.get("Image")
    image_data = None
    start_time = time.time()

    if image_path:
        try:
            async with aiofiles.open(image_path, 'rb') as f:
                image_data = await f.read()
        except Exception as e:
            logging.warning(f"无法读取图片 {image_path}: {str(e)}")
    result_template = {
        "question_id": question_id,
        "question": question,
        "subject_area": subject_area,  # 添加学科领域
        "historical_period": historical_period, 
        "core_purpose": core_purpose,
        "model_answer": None,
        "expected_answer": expected_answer,
        "answer_type": answer_type,
        "result": "未完成",
        "score": 0,
        "max_score": 2 if answer_type == "exactMatch" else 10,
        "assessment": None,
        "error": None,
        "processing_time": 0,
        "timestamp": datetime.now().isoformat()
    }

    try:
        logging.info(f"提问 [{question_id}] [{subject_area}]: {question}")
        max_retries = 5
        for attempt in range(1,max_retries+1):
            # 添加超时处理
            try:
                response = await asyncio.wait_for(agent.answer_question(question,image=image_data), timeout=timeout)
                if response:
                    response = response.strip()
                    logging.info(f"OpenAI模型回答: {response}")
                    result_template["model_answer"] = response
                    break
                else:
                    raise Exception("无响应")
            except asyncio.TimeoutError:
                logging.warning(f"问题 '{question}' 回答超时")
                last_exception = "超时"
            except Exception as e:
                logging.error(f"回答问题时出错: {str(e)}")
                last_exception = str(e)
            if attempt < max_retries:
                await asyncio.sleep(1)
        else:
            result_template["result"] = "请求失败"
            result_template["error"] = last_exception
            return result_template

        if answer_type == "exactMatch":
            if compare_answers(response, expected_answer):
                logging.info("回答正确")
                result_template["score"] = 2
                result_template["result"] = "正确"
            else:
                logging.info(f"回答错误，期望: {expected_answer}")
                result_template["result"] = f"错误，期望: {expected_answer}"
        elif answer_type == "shortAnswer":
            try:
                    # 添加评分超时处理
                try:
                    assessment = await asyncio.wait_for(
                        judging_agent.judge_answer(question, response, expected_answer),
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    logging.warning(f"问题 '{question}' 评分超时")
                    result_template["result"] = "评分超时"
                    return result_template

                logging.info(f"评分结果: {assessment}")
                result_template["assessment"] = assessment

                try:
                    parsed = _parse_score(assessment)
                    score = parsed if parsed is not None else 0
                except Exception:
                    score = 0
                result_template["score"] = score

                if score >= 7:
                    logging.info("回答基本正确")
                    result_template["result"] = "基本正确"
                else:
                    logging.info(f"回答不正确，期望: {expected_answer}")
                    result_template["result"] = f"不正确，期望: {expected_answer}"
            except Exception as e:
                logging.error(f"评分时出错: {str(e)}")
                result_template["result"] = "评分失败"
                result_template["error"] = str(e)
        else:
            logging.error("请求失败，无响应")
            result_template["result"] = "请求失败"

        return result_template
    except Exception as e:
        logging.error(f"评估问题时发生意外错误: {str(e)}")
        result_template["result"] = "处理失败"
        result_template["error"] = str(e)
        return result_template
    finally:
        # 记录处理时间
        result_template["processing_time"] = round(time.time() - start_time, 2)

# 执行测评
async def evaluate_model(qa_data, agent, judging_agent, csv_filename, batch_size=5, timeout=500):
    total_score = 0
    total_possible_score = sum(
        2 if qa.get("Answer Type", "exactMatch") == "exactMatch" else 10 for qa in qa_data
    )

    processed_count = 0
    batch_counter = 0
    results = []

    # 按学科领域分组统计
    subject_stats = {}
    historical_period_stats = {}
    core_purpose_stats = {}

    for i in range(0, len(qa_data), batch_size):
        batch_counter += 1
        batch = qa_data[i:i + batch_size]
        logging.info(f"开始处理批次 #{batch_counter} (问题 {i + 1}-{min(i + batch_size, len(qa_data))})")

        # 创建任务列表并添加超时
        tasks = []
        for qa in batch:
            task = asyncio.create_task(
                evaluate_single_question(qa, agent, judging_agent, timeout)
            )
            tasks.append(task)

        try:
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        except Exception as e:
            logging.error(f"处理批次时出错: {str(e)}")
            # 收集已完成的结果
            batch_results = []
            for task in tasks:
                if task.done():
                    batch_results.append(task.result())
                else:
                    qa = task.get_extra_info('qa', {}) if hasattr(task, 'get_extra_info') else {}
                    subject = qa.get("学科领域", "未知领域") if qa else "未知领域"
                    historical_period = qa.get("历史分期", "未知分期")
                    core_purpose = qa.get("核心目的", "未知目的")
                    batch_results.append({
                        "question_id": qa.get("id", "unknown") if qa else "unknown",
                        "question": "未知问题",
                        "subject_area": subject,
                        "model_answer": None,
                        "expected_answer": None,
                        "answer_type": "unknown",
                        "result": "批次处理失败",
                        "score": 0,
                        "max_score": 0,
                        "assessment": None,
                        "error": str(e),
                        "processing_time": 0,
                        "timestamp": datetime.now().isoformat()
                    })
            results.extend(batch_results)

        # 将本批次结果写入CSV
        for result in batch_results:
            processed_count += 1
            total_score += result['score']

            # 更新学科统计
            subject = result['subject_area']
            if subject not in subject_stats:
                subject_stats[subject] = {
                    "count": 0,
                    "score": 0,
                    "max_score": 0,
                    "processing_time": 0
                }

            subject_stats[subject]["count"] += 1
            subject_stats[subject]["score"] += result['score']
            subject_stats[subject]["max_score"] += result['max_score']
            subject_stats[subject]["processing_time"] += result['processing_time']
            
            historical_period = result['historical_period']
            if historical_period not in historical_period_stats:
                historical_period_stats[historical_period] = {
                    "count": 0,
                    "score": 0,
                    "max_score": 0,
                    "processing_time": 0
                }
            historical_period_stats[historical_period]["count"] += 1
            historical_period_stats[historical_period]["score"] += result['score']
            historical_period_stats[historical_period]["max_score"] += result['max_score']
            historical_period_stats[historical_period]["processing_time"] += result['processing_time']
            
            core_purpose = result['core_purpose']
            if core_purpose not in core_purpose_stats:
                core_purpose_stats[core_purpose] = {
                    "count": 0,
                    "score": 0,
                    "max_score": 0,
                    "processing_time": 0
                }
            core_purpose_stats[core_purpose]["count"] += 1
            core_purpose_stats[core_purpose]["score"] += result['score']
            core_purpose_stats[core_purpose]["max_score"] += result['max_score']
            core_purpose_stats[core_purpose]["processing_time"] += result['processing_time']
            # 准备CSV行数据
            csv_row = [
                result['question_id'],
                result['question'],
                result['subject_area'],
                result['model_answer'],
                result['expected_answer'],
                result['answer_type'],
                result['result'],
                result['score'],
                result['max_score'],
                result['assessment'],
                result['error'],
                result['processing_time'],
                result['timestamp']
            ]
            await append_to_csv(csv_filename, csv_row)

        # 进度报告
        progress = processed_count / len(qa_data) * 100
        logging.info(f"进度: {processed_count}/{len(qa_data)} ({progress:.1f}%)")
        logging.info(f"当前得分: {total_score}/{total_possible_score}")

        # 每批次后短暂暂停
        await asyncio.sleep(1)

    final_result = {
        "total_questions": len(qa_data),
        "total_score": total_score,
        "total_possible_score": total_possible_score,
        "accuracy": total_score / total_possible_score * 100 if total_possible_score > 0 else 0,
        "subject_stats": subject_stats,
        "historical_period_stats": historical_period_stats,
        "core_purpose_stats": core_purpose_stats
    }

    logging.info(
        f"测评完成: 总得分 {total_score}/{total_possible_score} "
        f"({final_result['accuracy']:.1f}%)"
    )

    return final_result, results

# 生成学科统计报告
def generate_subject_stats(subject_stats):
    if not subject_stats:
        return "无学科统计数据"

    report_lines = ["\n学科领域统计:"]
    report_lines.append(f"{'学科领域':<15} {'问题数':<8} {'得分':<8} {'总分':<8} {'准确率':<10} {'平均用时':<10}")
    report_lines.append("-" * 65)

    for subject, stats in subject_stats.items():
        count = stats["count"]
        score = stats["score"]
        max_score = stats["max_score"]
        avg_time = stats["processing_time"] / count if count > 0 else 0

        accuracy = (score / max_score * 100) if max_score > 0 else 0

        report_lines.append(
            f"{subject:<15} {count:<8} {score:<8} {max_score:<8} "
            f"{accuracy:.1f}%{'':<5} {avg_time:.2f}秒"
        )

    return "\n".join(report_lines)

# 生成总结报告
async def generate_summary_report(results, processed_files, elapsed_time):
    # 总体总结
    summary = f"""
============= 评测总结 =============
评测时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
处理文件: {", ".join(processed_files)}
总问题数: {results['total_questions']}
总得分: {results['total_score']}/{results['total_possible_score']}
准确率: {results['accuracy']:.1f}%
运行时间: {elapsed_time:.2f} 秒 ({elapsed_time / 60:.2f} 分钟)
"""

    # 添加学科统计
    if 'subject_stats' in results:
        summary += generate_subject_stats(results['subject_stats'])

    if 'historical_period_stats' in results:
        summary += "\n" + generate_subject_stats(results['historical_period_stats']).replace("学科领域", "历史分期")
    
    if 'core_purpose_stats' in results:
        summary += "\n" + generate_subject_stats(results['core_purpose_stats']).replace("学科领域", "核心目的")

    summary += "\n==================================\n"
    return summary

# 主函数
async def main():
    start_time = time.time()
    processed_files = []
    all_results = []
    csv_filename = get_csv_filename()

    try:
        # 初始化CSV文件
        if not await init_csv_file(csv_filename):
            logging.error("CSV文件初始化失败，终止程序")
            exit(1)
        logging.info(f"已创建结果文件: {csv_filename}")

        config1 = await load_config1()
        config2 = await load_config2()
        qa_files = ["评测/0828所有vqa简答.jsonl"]  # 可以扩展为多个文件

        agent = AnsweringAgent(config1)
        judging_agent = JudgingAgent(config2)

        summary_results = {
            "total_questions": 0,
            "total_score": 0,
            "total_possible_score": 0,
            "subject_stats": {}
        }

        for qa_file in qa_files:
            logging.info(f"开始处理文件: {qa_file}")
            qa_data = await load_qa_data(qa_file)
            processed_files.append(qa_file)

            file_results, detailed_results = await evaluate_model(
                qa_data, agent, judging_agent, csv_filename
            )
            all_results.extend(detailed_results)

            # 累加总结数据
            summary_results["total_questions"] += file_results["total_questions"]
            summary_results["total_score"] += file_results["total_score"]
            summary_results["total_possible_score"] += file_results["total_possible_score"]

            # 合并学科统计数据
            if 'subject_stats' in file_results:
                for subject, stats in file_results['subject_stats'].items():
                    if subject not in summary_results["subject_stats"]:
                        summary_results["subject_stats"][subject] = {
                            "count": 0,
                            "score": 0,
                            "max_score": 0,
                            "processing_time": 0
                        }
                    summary_results["subject_stats"][subject]["count"] += stats["count"]
                    summary_results["subject_stats"][subject]["score"] += stats["score"]
                    summary_results["subject_stats"][subject]["max_score"] += stats["max_score"]
                    summary_results["subject_stats"][subject]["processing_time"] += stats["processing_time"]

            if 'historical_period_stats' in file_results:
                if "historical_period_stats" not in summary_results:
                    summary_results["historical_period_stats"] = {}
                for period, stats in file_results['historical_period_stats'].items():
                    if period not in summary_results["historical_period_stats"]:
                        summary_results["historical_period_stats"][period] = {
                            "count": 0,
                            "score": 0,
                            "max_score": 0,
                            "processing_time": 0
                        }
                    summary_results["historical_period_stats"][period]["count"] += stats["count"]
                    summary_results["historical_period_stats"][period]["score"] += stats["score"]
                    summary_results["historical_period_stats"][period]["max_score"] += stats["max_score"]
                    summary_results["historical_period_stats"][period]["processing_time"] += stats["processing_time"]

            if 'core_purpose_stats' in file_results:
                if "core_purpose_stats" not in summary_results:
                    summary_results["core_purpose_stats"] = {}
                for purpose, stats in file_results['core_purpose_stats'].items():
                    if purpose not in summary_results["core_purpose_stats"]:
                        summary_results["core_purpose_stats"][purpose] = {
                            "count": 0,
                            "score": 0,
                            "max_score": 0,
                            "processing_time": 0
                        }
                    summary_results["core_purpose_stats"][purpose]["count"] += stats["count"]
                    summary_results["core_purpose_stats"][purpose]["score"] += stats["score"]
                    summary_results["core_purpose_stats"][purpose]["max_score"] += stats["max_score"]
                    summary_results["core_purpose_stats"][purpose]["processing_time"] += stats["processing_time"]
            # 文件间暂停
            await asyncio.sleep(2)

        # 计算总准确率
        if summary_results["total_possible_score"] > 0:
            summary_results["accuracy"] = (
                    summary_results["total_score"] / summary_results["total_possible_score"] * 100
            )
        else:
            summary_results["accuracy"] = 0.0

        # 生成并保存总结报告
        elapsed_time = time.time() - start_time
        summary_report = await generate_summary_report(summary_results, processed_files, elapsed_time)
        await export_summary(summary_report)
        logging.info(summary_report)

    except KeyboardInterrupt:
        logging.info("用户中断，正在保存已处理结果...")
        elapsed_time = time.time() - start_time

        # 尝试生成部分学科统计
        partial_subject_stats = {}
        for result in all_results:
            subject = result['subject_area']
            if subject not in partial_subject_stats:
                partial_subject_stats[subject] = {
                    "count": 0,
                    "score": 0,
                    "max_score": 0,
                    "processing_time": 0
                }
            partial_subject_stats[subject]["count"] += 1
            partial_subject_stats[subject]["score"] += result['score']
            partial_subject_stats[subject]["max_score"] += result['max_score']
            partial_subject_stats[subject]["processing_time"] += result['processing_time']

        partial_results = {
            "total_questions": len(all_results),
            "total_score": sum(r['score'] for r in all_results),
            "total_possible_score": sum(r['max_score'] for r in all_results),
            "accuracy": 0,
            "subject_stats": partial_subject_stats
        }

        if partial_results["total_possible_score"] > 0:
            partial_results["accuracy"] = (
                    partial_results["total_score"] / partial_results["total_possible_score"] * 100
            )

        partial_summary = await generate_summary_report(partial_results, processed_files, elapsed_time)
        partial_summary = "====== 部分结果 (用户中断) ======\n" + partial_summary

        await export_summary(partial_summary)
        logging.info(partial_summary)
    except Exception as e:
        logging.error(f"主程序发生错误: {str(e)}")
        elapsed_time = time.time() - start_time

        # 尝试生成部分学科统计
        partial_subject_stats = {}
        for result in all_results:
            subject = result['subject_area']
            if subject not in partial_subject_stats:
                partial_subject_stats[subject] = {
                    "count": 0,
                    "score": 0,
                    "max_score": 0,
                    "processing_time": 0
                }
            partial_subject_stats[subject]["count"] += 1
            partial_subject_stats[subject]["score"] += result['score']
            partial_subject_stats[subject]["max_score"] += result['max_score']
            partial_subject_stats[subject]["processing_time"] += result['processing_time']

        partial_results = {
            "total_questions": len(all_results),
            "total_score": sum(r['score'] for r in all_results),
            "total_possible_score": sum(r['max_score'] for r in all_results),
            "accuracy": 0,
            "subject_stats": partial_subject_stats
        }

        if partial_results["total_possible_score"] > 0:
            partial_results["accuracy"] = (
                    partial_results["total_score"] / partial_results["total_possible_score"] * 100
            )

        error_summary = await generate_summary_report(partial_results, processed_files, elapsed_time)
        error_summary = f"!!!!!!! 评测异常终止 !!!!!!!\n错误: {str(e)}\n" + error_summary

        await export_summary(error_summary)
    finally:
        elapsed_time = time.time() - start_time
        time_summary = f"总运行时间: {elapsed_time:.2f} 秒 ({elapsed_time / 60:.2f} 分钟)"
        logging.info(time_summary)


if __name__ == "__main__":
    try:
        # 设置更合理的事件循环策略
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy()
                                      if os.name == 'nt'
                                      else asyncio.DefaultEventLoopPolicy())

        # 运行主程序
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("程序被用户中断")
    except Exception as e:
        logging.error(f"程序意外终止: {str(e)}")