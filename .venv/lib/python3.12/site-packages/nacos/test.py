import time

# from gevent import monkey

import nacos

SERVER_ADDRESSES = "11.161.207.61:8848"
NAMESPACE = "20d99327-263e-47ae-b705-80a0515617b1"
USERNAME = 'test-user'
PASSWORD = 'test-user'


def test_cb(args):
    print(args)


def setup_nacos():

    client = nacos.NacosClient(SERVER_ADDRESSES, namespace=NAMESPACE, username=USERNAME, password=PASSWORD,
                               log_level='DEBUG')
    d = "test"
    g = "DEFAULT_GROUP"
    client.add_config_watcher(d, g, test_cb)
    time.sleep(5)
    client.publish_config(d, g, '12345', app_name='test', config_type='text')



def main():
    # monkey.patch_all()

    setup_nacos()
    time.sleep(5)

    try:
        # 您可以通过更复杂的方式来控制这个循环，这里只是简单地让它永远运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Test terminated by user")
    # time.sleep(10000)


if __name__ == '__main__':
    main()
