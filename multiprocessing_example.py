import multiprocessing


class Consumer(multiprocessing.Process):

    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            print('%s: %s' % (proc_name, next_task))

            try:
                if "fnc" in next_task:
                    answer = self.calculate(next_task)
                    self.result_queue.put(answer)
                    print("exiting")
                    self.task_queue.task_done()
                    break
            except Exception as e:
                print(e)

        #self.task_queue.task_done()
        return

    def calculate(self, function_dict: dict):
        function = function_dict["fnc"]
        params = function_dict["params"]
        return function(**params)


def fct_ex(text):
    return text


if __name__ == '__main__':
    # Establish communication queues
    tasks = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()
    # Start consumers
    num_consumers = multiprocessing.cpu_count()
    consumers = [Consumer(tasks, results) for i in range(num_consumers)]
    for w in consumers:
        w.start()
    # Enqueue jobs
    num_jobs = 16
    for i in range(num_jobs):
        tasks.put({"fnc": fct_ex, "params": {"text": "bla"}})
    # Add a poison pill for each consumer
    #for i in range(num_consumers):
    #    tasks.put(None)


    # Wait for all of the tasks to finish
    tasks.join()

    for w in consumers:
        w.terminate()

    # Start printing results
    while num_jobs:
        result = results.get()
        print(result)
        num_jobs -= 1
