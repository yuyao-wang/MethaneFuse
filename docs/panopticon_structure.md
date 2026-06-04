// Multi-Sensor Panopticon 4 Overall Boost
digraph {
	nodesep=0.5 rankdir=LR ranksep=1.0
	node [fontname=Arial fontsize=12]
	color=lightgrey label="Data Input (Wide-table Row)" style=filled
	row [label="Multi-Sensor Row
(S2, L89, S5P, WV3)" shape=box]
	collate [label="custom_collate_fn
(Sample Unrolling)" shape=ellipse]
	subgraph cluster_data {
	}
	color=lightyellow label="Core Model (Shared Backbone)" style=filled
	pe [label="Sensor-Specific PatchEmbed
(Adapters)" color=blue shape=trapezium]
	vit [label="DinoViT B/14 Backbone
(Shared Weights)" shape=rect style=bold]
	subgraph cluster_model {
	}
	s_heads [label="Individual Sensor Heads
(s2, l89, s5p, wv3)" color=blue shape=box]
	pooling [label="MAP Pooling
(Masked Attention)" color=purple shape=diamond]
	f_head [label="Row Fusion Head" color=purple shape=box]
	g_head [label="Row Gate Head" color=orange shape=box]
	ref_logits [label="Ref Row Logits
(Mean of Sample Logits)" shape=ellipse]
	gate_val [label="Gate (g)" color=orange shape=circle]
	final [label="Final Logits
(g*Fused + (1-g)*Ref)" color=gold shape=star style=filled]
	color=mistyrose label="Loss Definition (Total Loss)" style=filled
	l_main [label="L_fused_ce
(Weight: 1.0)" color=red shape=note]
	l_aux [label="L_sensor_aux
(Weight: 0.3)" color=red shape=note]
	l_single [label="L_single_fused
(Weight: 0.5)" color=red shape=note]
	l_gate [label="L_gate_bce
(Weight: 0.02)" color=red shape=note]
	subgraph cluster_loss {
	}
	row -> collate
	collate -> pe
	pe -> vit
	vit -> s_heads
	vit -> pooling
	pooling -> f_head
	pooling -> g_head
	s_heads -> ref_logits
	f_head -> final
	ref_logits -> final
	g_head -> gate_val
	gate_val -> final
	final -> l_main [color=red style=dashed]
	s_heads -> l_aux [color=red style=dashed]
	f_head -> l_single [label="if Single Sensor" color=red style=dashed]
	gate_val -> l_gate [color=red style=dashed]
}
