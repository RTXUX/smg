//! V4.1's mixed plain-text / added-token boundaries. Symbolic vocabulary
//! terminals are lowered by the gateway tokenizer before backend validation.
use openai_protocol::common::Tool;

use super::{CallMarkers, Count, Grammar};
use crate::tagged::attributed::Tags;

const DSML: &str = "｜DSML｜";

fn marker(text: &str) -> Grammar {
    let mut elements = Vec::new();
    for (index, text) in text.split(DSML).enumerate() {
        if index > 0 {
            elements.push(Grammar::TokenSpelling(DSML.into()));
        }
        if !text.is_empty() {
            elements.push(Grammar::ConstString(text.into()));
        }
    }
    Grammar::Sequence(elements)
}

pub(super) fn calls(
    markers: &CallMarkers<'_>,
    tags: &Tags,
    tools: &[&Tool],
    count: Count<'_>,
    reasoning_open: bool,
) -> Option<Grammar> {
    let block = markers.block.as_ref()?;
    let parameter = Grammar::Sequence(vec![
        marker(tags.parameter_open),
        Grammar::Regex("[^\"]+".into()),
        Grammar::Or(vec![
            Grammar::Sequence(vec![
                Grammar::TokenSpan {
                    prefix: "\" string=\"true\">".into(),
                    suffix: "</".into(),
                },
                marker(&format!("{}\n", tags.parameter_close.strip_prefix("</")?)),
            ]),
            Grammar::Sequence(vec![
                Grammar::ConstString("\" string=\"false\">".into()),
                Grammar::TokenJson,
                marker(&format!("{}\n", tags.parameter_close)),
            ]),
        ]),
    ]);
    let invokes = Grammar::Or(
        tools
            .iter()
            .map(|tool| {
                Grammar::Sequence(vec![
                    marker(&format!("{}{}\">\n", markers.call_open, tool.function.name)),
                    Grammar::Star(Box::new(parameter.clone())),
                    marker(&format!("{}\n", markers.call_close)),
                ])
            })
            .collect(),
    );
    let free = reasoning_open || !count.at_least_one;
    let begin = if free {
        Grammar::Sequence(vec![
            Grammar::TokenSuffix("<".into()),
            marker(&format!("{}\n", block.open.strip_prefix('<')?)),
        ])
    } else {
        marker(&format!("{}{}\n", tags.before_block, block.open))
    };
    let block = Grammar::Sequence(vec![
        begin,
        count.calls(invokes),
        marker(&count.end(block.close)),
    ]);
    let block = if count.at_least_one {
        block
    } else {
        Grammar::Optional(Box::new(block))
    };
    Some(if free {
        Grammar::Sequence(vec![Grammar::AnyTokens(vec![DSML.into()]), block])
    } else {
        block
    })
}
