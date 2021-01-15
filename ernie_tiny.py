from paddlehub.serving.bert_serving import bs_client

bc = bs_client.BSClient(module_name="ernie_tiny", server="10.1.12.33:8866")
input_text = [["西风吹老洞庭波"], ["一夜湘君白发多"], ["醉后不知天在水"], ["满船清梦压星河"], ]
result = bc.get_result(input_text=input_text)
