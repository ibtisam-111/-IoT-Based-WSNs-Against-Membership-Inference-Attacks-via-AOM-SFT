import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class PrivacyMetrics:
    def __init__(self):
        self.metrics_history = []
    
    def compute_mia_metrics(self, y_true, y_pred, y_prob=None):
        metrics = {}
        
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        if y_prob is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
            except:
                metrics['auc_roc'] = 0.0
        
        # Attack advantage (True Positive Rate - False Positive Rate)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['attack_advantage'] = tpr - fpr
        
        # Random guessing baseline comparison
        metrics['improvement_over_random'] = metrics['accuracy'] - 0.5
        
        self.metrics_history.append(metrics)
        return metrics
    
    def compute_utility_metrics(self, y_true, y_pred, original_accuracy=None):
        metrics = {}
        
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        if original_accuracy is not None:
            metrics['accuracy_drop'] = original_accuracy - metrics['accuracy']
            metrics['utility_preservation'] = (metrics['accuracy'] / original_accuracy) * 100
        
        return metrics
    
    def compute_privacy_utility_tradeoff(self, privacy_metrics, utility_metrics):
        tradeoff_metrics = {}
        
        # Privacy score (lower is better for defense)
        privacy_score = privacy_metrics.get('accuracy', 0.5)
        
        # Utility score (higher is better)
        utility_score = utility_metrics.get('accuracy', 0.0)
        
        # Combined score (balanced metric)
        if privacy_score <= 0.5:
            # Good privacy protection
            tradeoff_metrics['balanced_score'] = utility_score
        else:
            # Poor privacy protection, penalize
            tradeoff_metrics['balanced_score'] = utility_score * (1 - privacy_score)
        
        tradeoff_metrics['privacy_score'] = privacy_score
        tradeoff_metrics['utility_score'] = utility_score
        
        return tradeoff_metrics

class ConfidenceMetrics:
    def __init__(self):
        self.confidence_stats = []
    
    def analyze_confidence_distribution(self, member_confidences, nonmember_confidences):
        stats = {}
        
        stats['member_mean_confidence'] = np.mean(member_confidences)
        stats['nonmember_mean_confidence'] = np.mean(nonmember_confidences)
        stats['confidence_gap'] = stats['member_mean_confidence'] - stats['nonmember_mean_confidence']
        
        stats['member_std_confidence'] = np.std(member_confidences)
        stats['nonmember_std_confidence'] = np.std(nonmember_confidences)
        
        # KS test for distribution difference
        from scipy import stats as scipy_stats
        ks_stat, ks_pvalue = scipy_stats.ks_2samp(member_confidences, nonmember_confidences)
        stats['ks_statistic'] = ks_stat
        stats['ks_pvalue'] = ks_pvalue
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((stats['member_std_confidence']**2 + stats['nonmember_std_confidence']**2) / 2)
        stats['effect_size'] = stats['confidence_gap'] / pooled_std if pooled_std > 0 else 0
        
        self.confidence_stats.append(stats)
        return stats
    
    def compute_confidence_overlap(self, member_confidences, nonmember_confidences, bins=20):
        hist_member, bin_edges = np.histogram(member_confidences, bins=bins, density=True)
        hist_nonmember, _ = np.histogram(nonmember_confidences, bins=bin_edges, density=True)
        
        overlap = np.sum(np.minimum(hist_member, hist_nonmember)) * (bin_edges[1] - bin_edges[0])
        return overlap

class DefenseEffectiveness:
    def __init__(self):
        self.effectiveness_metrics = []
    
    def compute_defense_effectiveness(self, attack_results_no_defense, attack_results_with_defense):
        effectiveness = {}
        
        # Attack success rate reduction
        asr_no_defense = attack_results_no_defense.get('overall_accuracy', 0.5)
        asr_with_defense = attack_results_with_defense.get('overall_accuracy', 0.5)
        
        effectiveness['asr_reduction'] = asr_no_defense - asr_with_defense
        effectiveness['asr_reduction_percentage'] = (effectiveness['asr_reduction'] / asr_no_defense) * 100
        
        # Utility preservation
        utility_no_defense = attack_results_no_defense.get('utility_accuracy', 1.0)
        utility_with_defense = attack_results_with_defense.get('utility_accuracy', 1.0)
        
        effectiveness['utility_preservation'] = utility_with_defense / utility_no_defense
        
        # Privacy-utility score
        effectiveness['privacy_utility_score'] = (
            effectiveness['asr_reduction'] * effectiveness['utility_preservation']
        )
        
        self.effectiveness_metrics.append(effectiveness)
        return effectiveness

class StatisticalTests:
    @staticmethod
    def significance_test(metric_a, metric_b, test_type='t_test'):
        from scipy import stats
        
        if test_type == 't_test':
            t_stat, p_value = stats.ttest_ind(metric_a, metric_b)
            return {'t_statistic': t_stat, 'p_value': p_value}
        
        elif test_type == 'mannwhitney':
            u_stat, p_value = stats.mannwhitneyu(metric_a, metric_b)
            return {'u_statistic': u_stat, 'p_value': p_value}
        
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
    
    @staticmethod
    def confidence_interval(data, confidence=0.95):
        import scipy.stats as st
        
        n = len(data)
        mean = np.mean(data)
        sem = st.sem(data)
        
        if n < 30:
            # Use t-distribution for small samples
            interval = st.t.interval(confidence, n-1, loc=mean, scale=sem)
        else:
            # Use normal distribution for large samples
            interval = st.norm.interval(confidence, loc=mean, scale=sem)
        
        return {
            'mean': mean,
            'lower_bound': interval[0],
            'upper_bound': interval[1],
            'margin_of_error': (interval[1] - interval[0]) / 2
        }

class VisualizationMetrics:
    @staticmethod
    def plot_confidence_distributions(member_confidences, nonmember_confidences, defense_name=""):
        plt.figure(figsize=(10, 6))
        
        plt.hist(member_confidences, bins=30, alpha=0.7, label='Member', density=True)
        plt.hist(nonmember_confidences, bins=30, alpha=0.7, label='Non-member', density=True)
        
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Density')
        plt.title(f'Confidence Distributions - {defense_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
    
    @staticmethod
    def plot_privacy_utility_tradeoff(privacy_scores, utility_scores, defense_names):
        plt.figure(figsize=(10, 8))
        
        # Ideal point (low privacy risk, high utility)
        plt.scatter([0.5], [1.0], color='green', marker='*', s=200, label='Ideal', zorder=5)
        
        # Plot each defense
        for i, (privacy, utility, name) in enumerate(zip(privacy_scores, utility_scores, defense_names)):
            plt.scatter(privacy, utility, s=100, label=name)
            plt.annotate(name, (privacy, utility), xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Privacy Risk (MIA Accuracy)')
        plt.ylabel('Utility (Task Accuracy)')
        plt.title('Privacy-Utility Tradeoff')
        plt.xlim(0.45, 0.85)
        plt.ylim(0.7, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        return plt.gcf()
    
    @staticmethod
    def plot_attack_success_comparison(attack_results_dict):
        defenses = list(attack_results_dict.keys())
        success_rates = [attack_results_dict[defense]['overall_accuracy'] for defense in defenses]
        
        plt.figure(figsize=(12, 6))
        
        bars = plt.bar(defenses, success_rates, color=['red' if rate > 0.6 else 'orange' for rate in success_rates])
        plt.axhline(y=0.5, color='green', linestyle='--', label='Random Guessing')
        
        plt.ylabel('MIA Success Rate')
        plt.title('Membership Inference Attack Success Rates by Defense')
        plt.xticks(rotation=45)
        plt.legend()
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{rate:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return plt.gcf()

def compute_comprehensive_metrics(y_true_mia, y_pred_mia, y_prob_mia, 
                                 y_true_task, y_pred_task, original_accuracy):
    privacy_metrics = PrivacyMetrics()
    utility_metrics = PrivacyMetrics()  # Reusing for utility
    
    mia_metrics = privacy_metrics.compute_mia_metrics(y_true_mia, y_pred_mia, y_prob_mia)
    task_metrics = utility_metrics.compute_utility_metrics(y_true_task, y_pred_task, original_accuracy)
    
    tradeoff_metrics = privacy_metrics.compute_privacy_utility_tradeoff(mia_metrics, task_metrics)
    
    comprehensive_results = {
        'privacy_metrics': mia_metrics,
        'utility_metrics': task_metrics,
        'tradeoff_metrics': tradeoff_metrics
    }
    
    return comprehensive_results

# Example usage
if __name__ == "__main__":
    # Test metrics with dummy data
    np.random.seed(42)
    
    # Simulate MIA results
    y_true_mia = np.random.randint(0, 2, 1000)
    y_pred_mia = np.random.randint(0, 2, 1000)
    y_prob_mia = np.random.random(1000)
    
    # Simulate task results
    y_true_task = np.random.randint(0, 3, 1000)
    y_pred_task = np.random.randint(0, 3, 1000)
    
    metrics = compute_comprehensive_metrics(
        y_true_mia, y_pred_mia, y_prob_mia,
        y_true_task, y_pred_task, original_accuracy=0.95
    )
    
    print("Comprehensive Metrics Results:")
    for category, results in metrics.items():
        print(f"\n{category.upper()}:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")
