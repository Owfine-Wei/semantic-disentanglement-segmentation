def calculate_loss_averages(file_path):
    total_loss_img = 0.0
    total_consist_loss = 0.0
    total_loss_origin = 0.0
    count = 0

    try:
        with open(file_path, 'r') as f:
            for line in f:
                # 1. 移除换行符并按逗号分割
                parts = line.strip().split(',')
                if len(parts) != 3:
                    continue  # 跳过不符合格式的行
                
                # 2. 提取数值
                # 这里假设格式固定为 "标签数值"，通过替换掉标签文字来提取
                l_img = float(parts[0].replace('loss_img', ''))
                l_consist = float(parts[1].replace('loss_consist', ''))
                l_origin = float(parts[2].replace('loss_origin', ''))
                
                # 3. 累加
                total_loss_img += l_img
                total_consist_loss += l_consist
                total_loss_origin += l_origin
                count += 1

        if count > 0:
            # 4. 计算均值
            avg_img = total_loss_img / (count+1e-6)
            avg_consist = total_consist_loss / (count+1e-6)
            avg_origin = total_loss_origin / (count+1e-6)

            print(f"处理完成，共读取 {count} 行数据：")
            print("-" * 30)
            print(f"Loss Img 均值:     {avg_img:.6f}")
            print(f"Consist Loss 均值: {avg_consist:.6f}")
            print(f"Loss Origin 均值:  {avg_origin:.6f}")
        else:
            print("文件中没有有效数据。")

    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'")
    except Exception as e:
        print(f"运行出错: {e}")

# 调用脚本
calculate_loss_averages('./test.txt')