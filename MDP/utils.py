import numpy as np


# na: number of actions
def argmax(env, V, pi, s, gamma, na=4):
    e = np.zeroes(na)
    for a in range(na):
        q = 0
        P = 


def run_episode(env, policy, gamma, render = True):
	obs = env.reset()
	total_reward = 0
	step_idx = 0
	while True:
		if render:
			env.render()
		obs, reward, done , _ = env.step(int(policy[obs]))
		total_reward += (gamma ** step_idx * reward)
		step_idx += 1
		if done:
			break
	return total_reward

def evaluate_policy(env, policy, gamma , n = 100):
	scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
	return np.mean(scores)

def extract_policy(env,v, gamma):
	policy = np.zeros(env.nS)
	for s in range(env.nS):
		q_sa = np.zeros(env.nA)
		for a in range(env.nA):
			q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.P[s][a]])
		policy[s] = np.argmax(q_sa)
	return policy

def compute_policy_v(env, policy, gamma):
	v = np.zeros(env.nS)
	eps = 1e-5
	while True:
		prev_v = np.copy(v)
		for s in range(env.nS):
			policy_a = policy[s]
			v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, is_done in env.P[s][policy_a]])
		if (np.sum((np.fabs(prev_v - v))) <= eps):
			break
	return v

def policy_iteration(env, gamma):
	policy = np.random.choice(env.nA, size=(env.nS))  
	max_iters = 200000
	desc = env.unwrapped.desc
	for i in range(max_iters):
		old_policy_v = compute_policy_v(env, policy, gamma)
		new_policy = extract_policy(env,old_policy_v, gamma)
		#if i % 2 == 0:
		#	plot = plot_policy_map('Frozen Lake Policy Map Iteration '+ str(i) + 'Gamma: ' + str(gamma),new_policy.reshape(4,4),desc,colors_lake(),directions_lake())
		#	a = 1
		if (np.all(policy == new_policy)):
			k=i+1
			break
		policy = new_policy
	return policy,k

def value_iteration(env, gamma):
	v = np.zeros(env.nS)  # initialize value-function
	max_iters = 100000
	eps = 1e-20
	desc = env.unwrapped.desc
	for i in range(max_iters):
		prev_v = np.copy(v)
		for s in range(env.nS):
			q_sa = [sum([p*(r + gamma*prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)] 
			v[s] = max(q_sa)
		#if i % 50 == 0:
		#	plot = plot_policy_map('Frozen Lake Policy Map Iteration '+ str(i) + ' (Value Iteration) ' + 'Gamma: '+ str(gamma),v.reshape(4,4),desc,colors_lake(),directions_lake())
		if (np.sum(np.fabs(prev_v - v)) <= eps):
			k=i+1
			break
	return v,k

def plot_policy_map(title, policy, map_desc, color_map, direction_map):
	fig = plt.figure()
	ax = fig.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
	font_size = 'x-large'
	if policy.shape[1] > 16:
		font_size = 'small'
	plt.title(title)
	for i in range(policy.shape[0]):
		for j in range(policy.shape[1]):
			y = policy.shape[0] - i - 1
			x = j
			p = plt.Rectangle([x, y], 1, 1)
			p.set_facecolor(color_map[map_desc[i,j]])
			ax.add_patch(p)

			text = ax.text(x+0.5, y+0.5, direction_map[policy[i, j]], weight='bold', size=font_size,
						   horizontalalignment='center', verticalalignment='center', color='w')
			

	plt.axis('off')
	plt.xlim((0, policy.shape[1]))
	plt.ylim((0, policy.shape[0]))
	plt.tight_layout()
	plt.savefig(title+str('.png'))
	plt.close()

	return (plt)