import concurrent
import queue
import threading
from multiprocessing import pool


class ConcurrencyAdapter:

    def create_thread(self, target, *args, **kwargs):
        raise NotImplementedError()

    def create_thread_pool(self, size):
        raise NotImplementedError()

    def create_queue(self):
        raise NotImplementedError()


class NativeThreadAdapter(ConcurrencyAdapter):
    def create_thread(self, target, *args, **kwargs):
        thread = threading.Thread(target=target, args=args, kwargs=kwargs)
        thread.daemon = True
        return thread

    def create_queue(self):
        return queue.Queue()

    def create_thread_pool(self, size):
        return pool.ThreadPool(size)


class GeventAdapter(ConcurrencyAdapter):

    def create_thread(self, target, *args, **kwargs):
        from gevent import spawn
        return spawn(target, *args, **kwargs)

    def create_thread_pool(self, size):
        from gevent.pool import Pool
        return Pool(size)

    def create_queue(self):
        from gevent.queue import Queue
        return Queue()
