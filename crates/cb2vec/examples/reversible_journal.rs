use cb2vec::{ReversibleTokenJournal, TokenDelta, TokenDeltaReplay, TokenDeltaSink};

#[derive(Default)]
struct NumericState {
    values: [i32; 3],
}

impl TokenDeltaSink<u16> for NumericState {
    fn apply_site(&mut self, site: u16, deltas: &[TokenDelta<u16>], replay: TokenDeltaReplay) {
        match replay {
            TokenDeltaReplay::Forward => {
                for delta in deltas {
                    self.values[site as usize] +=
                        i32::from(delta.new_token()) - i32::from(delta.old());
                }
            }
            TokenDeltaReplay::Reverse => {
                for delta in deltas.iter().rev() {
                    self.values[site as usize] +=
                        i32::from(delta.old()) - i32::from(delta.new_token());
                }
            }
        }
    }
}

fn main() {
    let initial = [[1u16, 2], [3, 4], [5, 6]];
    let after = [[7u16, 2], [3, 4], [5, 8]];
    let mut journal = ReversibleTokenJournal::<u16, 2, 4>::new(3, 8);
    let mut state = NumericState::default();

    journal.reset(&initial);
    assert_eq!(journal.push_after(&after, &[0, 2]), 2);
    journal.materialize_pending(&mut state);
    assert_eq!(state.values, [6, 0, 2]);

    let popped = journal.pop(&mut state).expect("one frame");
    assert!(popped.was_materialized());
    assert_eq!(journal.logical_tokens(), initial);
    assert_eq!(state.values, [0, 0, 0]);
}
