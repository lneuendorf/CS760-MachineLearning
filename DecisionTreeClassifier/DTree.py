import math

class Node:
    # constuctor
    def __init__(self, df=None, feature_list=None, label_name=None, root=False, internal=False, 
                 leaf=False, label=None, split_feature=None, split_value=None, left=None, right=None):
        self.df = df
        self.feature_list = feature_list
        self.label_name = label_name
        self.root = root
        self.internal = internal
        self.leaf = leaf
        self.label = label
        self.split_feature = split_feature
        self.split_value = split_value
        self.left = left
        self.right = right

class DecisionTreeClassifier():
    def __init__(self):
        self.root = None
        self.dfs_preorder = []
    
    def train(self, df, feature_list, label_name):
        self.root = Node(df, feature_list, label_name, root=True)
        self.make_subtree(self.root)
        return self.root

    def make_subtree(self, node):
        c = self.get_candidate_splits(node.df, node.feature_list)
        if self.stopping_criteria_met(node.df, c, node.label_name):
            # make leaf node N
            node.leaf, node.internal = True, False
            # determine class label for N, whenever there is no majority class in leaf, predict y = 1
            if len(node.df[node.df[node.label_name]==1]) >=len(node.df[node.df[node.label_name]==0]):
                node.label = 1
            else:
                node.label = 0
            # add node to traversal list
            if node.root:
                self.dfs_preorder.append('(root)Y=%.0f' % node.label)
            else:
                self.dfs_preorder.append('(leaf)Y=%.0f' % node.label)
        else:
            split_feature, split_val = self.find_best_split(node.df, c, node.label_name)
            # add node to traversal list
            if node.root:
                self.dfs_preorder.append('(root)%s>=%.3f' % (split_feature,split_val))
            else:
                self.dfs_preorder.append('(internal)%s>=%.3f' % (split_feature,split_val))

            node.split_feature, node.split_value = split_feature, split_val
            # left branch gte candidate split
            left_node = Node(df=node.df[node.df[split_feature]>=split_val].reset_index(drop=True), 
                             feature_list=node.feature_list, label_name=node.label_name, internal=True) 
            node.left = left_node
            self.make_subtree(left_node)
            # right branch lt candidate split
            right_node = Node(df=node.df[node.df[split_feature]<split_val].reset_index(drop=True),
                              feature_list=node.feature_list, label_name=node.label_name, internal=True) 
            node.right = right_node
            self.make_subtree(right_node)

    def entropy(self, df, feature_name, split_val):
        pr_x_gte = len(df[df[feature_name]>=split_val]) / len(df)
        pr_x_lt = len(df[df[feature_name]<split_val]) / len(df)
        entorpy=0
        if pr_x_gte == 0:
            entropy = pr_x_lt * math.log2(pr_x_lt)
        elif pr_x_lt == 0:
            entropy = pr_x_gte * math.log2(pr_x_gte)
        else:
            entropy = (pr_x_gte * math.log2(pr_x_gte)) + (pr_x_lt * math.log2(pr_x_lt))
        return -entropy

    def conditional_entropy(self, df, feature_name, split_val, label_name):
        conditional_entropy = 0
        for label in df[label_name].unique():
            #when x>=split_val
            pr_xy = len(df[(df[label_name]==label) & (df[feature_name]>=split_val)]) / len(df)
            pr_x = len(df[df[feature_name]>=split_val]) / len(df)
            if (pr_xy>0) and (pr_x>0):
                conditional_entropy += (pr_xy * math.log2(pr_xy/pr_x))
            #when x<split_val
            pr_xy = len(df[(df[label_name]==label) & (df[feature_name]<split_val)]) / len(df)
            pr_x = len(df[df[feature_name]<split_val]) / len(df)
            if (pr_xy>0) and (pr_x>0):
                conditional_entropy += (pr_xy * math.log2(pr_xy/pr_x))
        return -conditional_entropy

    def info_gain(self, df, feature_name, split_val, label_name):
        return self.entropy(df, label_name, 1) - self.conditional_entropy(df, feature_name, split_val, label_name)

    def gain_ratio(self, df, feature_name, split_val, label_name):
        return self.info_gain(df, feature_name, split_val, label_name) / self.entropy(df, feature_name, split_val)
   
    def get_candidate_splits(self, df, feature_list):
        C = []
        for feature in feature_list:
            split_values = []
            for i in range(0,len(df[feature])):
                if df[feature][i] != df[feature].min(): # min val split candidate has zero entropy
                    if df[feature][i] not in split_values:
                        split_values.append(df[feature][i])
                        C.append([feature, df[feature][i]])
        return C
    
    def stopping_criteria_met(self, df, candidate_splits, label_name):    
        # (1) node is empty or entropy is zero (if only 1 sample)
        if len(df) <= 1:
            return True
        # (2) entropy of any candidate splits is zero -> occurs when all labels are the same
        for split in candidate_splits:
            if self.entropy(df, feature_name=split[0], split_val=split[1]) == 0:
                return True
        # (3) all splits have zero gain ratio
        best_split = self.find_best_split(df, candidate_splits, label_name)
        if self.gain_ratio(df, feature_name=best_split[0], split_val=best_split[1], label_name=label_name) == 0:
            return True
        # no stopping criteria met
        return False
    
    # returns in format split_feature_name, split_value
    def find_best_split(self, df, candidate_splits, label_name):
        best_split = candidate_splits[0]
        best_gain_ratio = self.gain_ratio(df, feature_name=best_split[0], split_val=best_split[1], label_name=label_name)
        # loop through candidate splits
        for split in candidate_splits:
            if self.entropy(df, feature_name=split[0], split_val=split[1]) != 0:
                gain_ratio = self.gain_ratio(df, feature_name=split[0], split_val=split[1], label_name=label_name)
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_split = split
        return best_split[0], best_split[1]

    def print_tree(self):
        print(self.dfs_preorder) 

    # appends y_hat predictions to corresponding dataframe row sample
    def predict(self, df_predict):
        df_predict['Y_hat'] = None
        for i in range(0,len(df_predict)):
            node = self.root
            sample = df_predict.iloc[i]
            while node.leaf == False:
                if sample[node.split_feature] >= node.split_value:
                    #print('>=')
                    node = node.left
                else:
                    #print('<')
                    node = node.right
            #print(node.label)
            df_predict['Y_hat'][i] = node.label
        return df_predict

    # pass in dataframe returned from predict method
    def error_rate(self, df, label_name, pred_name):
        return len(df[df[label_name] != df[pred_name]])/len(df)

        
