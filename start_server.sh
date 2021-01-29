hub serving start --config server_config.json &
#hub serving start bert_service -m ernie_tiny --use_gpu --gpu 0 --port 8867 &
#hub serving start -m senta_lstm --port 8866
#hub serving start -m vgg11_imagenet --port 8866
#hub serving start -m yolov3_darknet53_coco2017 --port 8866
