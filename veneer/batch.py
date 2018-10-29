
from .utils import DeferredActionCollection


class BatchRunner(object):
    '''
    Run multiple simulations and post-process summary results
    '''

    def __init__(self, endpoints):
        self.parameters = DeferredActionCollection('$', 'v.')
        self.parameters.model.defer()
        self.endpoints = endpoints

        self._retrieval = DeferredActionCollection('$', 'v.')
#    self._retrieval.instructions.append('result={}')

    def _run_single(self, parameter_set, endpoint, **kwargs):
        self.parameters.eval_script(parameter_set, {'v': endpoint})
        endpoint.drop_all_runs()
        ticket = endpoint.run_model(async=True, **kwargs)
        return (ticket, endpoint)

    def retrieve(self, var_name='y'):
        self._retrieval.instruction_prefix = "results['%s'] = v." % var_name
        return self._retrieval

    def _retrieve(self, ticket, endpoint):
        resp = ticket.getresponse()
        results = {}
        self._retrieval.eval_script({}, {'v': endpoint, 'results': results})
        return results

    def run(self, parameter_sets, **kwargs):
        if not len(self.endpoints):
            raise Exception('No model runners available')

        self.parameters.model.flush()

        available_endpoints = self.endpoints[:]
        jobs = []
        results = [None] * len(parameter_sets)

        for ix, row in parameter_sets.iterrows():
            print(ix, row)
            if len(available_endpoints) == 0:
                # Wait for something to be finished
                for i, (jx, row, ticket, endpoint) in enumerate(jobs):
                    if not ticket:
                        continue
                    results[i] = self._retrieve(ticket, endpoint)
                    # print(ix,row,results[i])
                    # print('Recycling %d',endpoint.port)
                    available_endpoints.append(endpoint)
                    jobs[i] = (jx, row, None, endpoint)
                    break

            endpoint_to_use = available_endpoints.pop()

            ticket, endpoint = self._run_single(row, endpoint_to_use, **kwargs)
            jobs.append((ix, row, ticket, endpoint))

        for i, (jx, row, ticket, endpoint) in enumerate(jobs):
            if not ticket:
                continue
            results[i] = self._retrieve(ticket, endpoint)
        return jobs, results
