//! Simple Gomocup protocol interface implementation.

use std::{io, time::Duration};

use figrid_board::Coord;
use figrid_board::legacy::{
    CaroEvaluator15, CaroEvaluator20, CaroSerCalc, Eval, FreeEvaluator15, FreeEvaluator20,
    FreeSerCalc, Rec, StdEvaluator15, StdEvaluator20, StdSerCalc,
};

struct ProtocolInfo {
    timeout_turn: i32,
    timeout_match: i32,
    time_left: i32,
    max_memory: Option<usize>,
    rule_exact5: bool,
    rule_continuous: bool,
    rule_renju: bool,
    rule_caro: bool,
}

impl ProtocolInfo {
    pub fn new() -> Self {
        Self {
            timeout_turn: 30000,
            timeout_match: 1000000000,
            time_left: 1000000000,
            max_memory: None,
            rule_exact5: false,
            rule_continuous: false,
            rule_renju: false,
            rule_caro: false,
        }
    }

    pub fn rule_supported(&self) -> bool {
        !(self.rule_continuous || self.rule_renju || (self.rule_caro && !self.rule_exact5))
    }

    pub fn ram_max(&self) -> usize {
        self.max_memory.unwrap_or(512 * 1024 * 1024)
    }

    pub fn timeout_turn<const SZ: usize>(&self, rec: &dyn Rec<SZ>) -> Duration {
        // XXX: making improvement here may improve the overall performance significantly.
        let timeout_turn = if rec.len() <= 5 {
            self.timeout_turn
        } else {
            self.timeout_turn
                .min(self.timeout_match / (SZ * SZ / 2) as i32)
                .min(self.time_left.max(0) / ((SZ * SZ - rec.len()) as i32 / 2).max(1))
        };
        Duration::from_millis(timeout_turn as u64)
    }

    pub fn update(&mut self, key: &str, val: &str) {
        let val = val.trim();
        match key {
            "timeout_turn" => self.timeout_turn = val.parse().unwrap(),
            "timeout_match" => self.timeout_match = val.parse().unwrap(),
            "max_memory" => {
                let max_memory: usize = val.parse().unwrap();
                if max_memory == 0 {
                    self.max_memory = None;
                } else {
                    self.max_memory.replace(max_memory);
                }
            }
            "time_left" => self.time_left = val.parse().unwrap(),
            "rule" => {
                let rule_byte: u8 = val.parse().unwrap();
                self.rule_exact5 = (rule_byte & 1) != 0;
                self.rule_continuous = (rule_byte & 2) != 0;
                self.rule_renju = (rule_byte & 4) != 0;
                self.rule_caro = (rule_byte & 8) != 0;
            }
            _ => (),
        }
    }
}

fn main() {
    std::thread::Builder::new()
        .stack_size(2 * 1024 * 1024)
        .spawn(main_loop)
        .unwrap()
        .join()
        .unwrap()
}

fn main_loop() {
    let mut is_size_20 = false;
    let mut info = ProtocolInfo::new();

    let mut eval_15: Option<Box<dyn Eval<15>>> = None;
    let mut eval_20: Option<Box<dyn Eval<20>>> = None;

    macro_rules! eval_switch_init {
        () => {
            let timeout = Duration::from_millis(info.timeout_turn as u64);
            if !info.rule_exact5 {
                if is_size_20 {
                    eval_20.replace(Box::new(FreeEvaluator20::new(FreeSerCalc::<20>, timeout))
                        as Box<dyn Eval<20>>);
                } else {
                    eval_15.replace(Box::new(FreeEvaluator15::new(FreeSerCalc::<15>, timeout))
                        as Box<dyn Eval<15>>);
                }
            } else if !info.rule_caro {
                if is_size_20 {
                    eval_20.replace(Box::new(StdEvaluator20::new(StdSerCalc::<20>, timeout))
                        as Box<dyn Eval<20>>);
                } else {
                    eval_15.replace(Box::new(StdEvaluator15::new(StdSerCalc::<15>, timeout))
                        as Box<dyn Eval<15>>);
                }
            } else {
                if is_size_20 {
                    eval_20.replace(Box::new(CaroEvaluator20::new(CaroSerCalc::<20>, timeout))
                        as Box<dyn Eval<20>>);
                } else {
                    eval_15.replace(Box::new(CaroEvaluator15::new(CaroSerCalc::<15>, timeout))
                        as Box<dyn Eval<15>>);
                }
            }
        };
    }

    loop {
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            std::process::exit(0);
        }
        let mut split_input = input.split_whitespace();
        let Some(command) = split_input.next() else {
            continue;
        };
        let command = command.trim().to_uppercase();
        match command.as_str() {
            "START" => {
                let Some(size) = split_input.next().and_then(|s| s.trim().parse::<u8>().ok())
                else {
                    println!("ERROR - cannot parse board size");
                    continue;
                };
                match size {
                    15 => is_size_20 = false,
                    20 => is_size_20 = true,
                    _ => {
                        println!("ERROR - unsupported board size");
                        continue;
                    }
                }
                if !info.rule_supported() {
                    println!("ERROR - unsupported rule");
                    continue;
                }
                eval_switch_init!();
                let ram_max = info.ram_max();
                if is_size_20 {
                    eval_20.as_mut().unwrap().set_max_ram(ram_max);
                } else {
                    eval_15.as_mut().unwrap().set_max_ram(ram_max);
                }
                println!("OK");
            }
            "TURN" | "BEGIN" => {
                let coord_x_y = if command.as_str() == "TURN" {
                    let mut str_coords = split_input.next().unwrap().split(',');
                    Some((
                        str_coords.next().unwrap().parse().unwrap(),
                        str_coords.next().unwrap().parse().unwrap(),
                    ))
                } else {
                    None
                };
                macro_rules! process_turn_begin {
                    ($eval_name:ident) => {
                        let eval = $eval_name.as_mut().unwrap();
                        if command.as_str() == "TURN" {
                            let (x, y) = coord_x_y.unwrap();
                            eval.add(Coord::from(x, y)).unwrap();
                        }
                        let timeout = info.timeout_turn((&eval).as_ref());
                        eval.set_turn_timeout(timeout);
                        let coord = eval.place_next();
                        if let Some((x, y)) = coord.get() {
                            println!("{x},{y}");
                        }
                    };
                }
                if is_size_20 {
                    process_turn_begin!(eval_20);
                } else {
                    process_turn_begin!(eval_15);
                }
            }
            "BOARD" => {
                macro_rules! process_board {
                    ($eval_name:ident) => {
                        let mut coords_own = Vec::new();
                        let mut coords_opp = Vec::new();
                        loop {
                            let mut input = String::new();
                            if io::stdin().read_line(&mut input).is_err() {
                                std::process::exit(0);
                            }
                            if input.to_uppercase().contains("DONE") {
                                break;
                            }
                            let mut nums: Vec<u8> = Vec::new();
                            for input_num in input.split(',') {
                                if let Ok(num) = input_num.trim().parse() {
                                    nums.push(num);
                                }
                            }
                            if nums.len() < 3 {
                                println!("ERROR - expected input: 'x,x,x'");
                                continue;
                            }
                            let coord = Coord::from(nums[0], nums[1]);
                            if nums[2] == 1 {
                                coords_own.push(coord);
                            } else if nums[2] == 2 {
                                coords_opp.push(coord);
                            }
                        }
                        let own_is_white =
                            (coords_own.len() as i32 + 1 - coords_opp.len() as i32) % 2 == 0;
                        let (coords_b, coords_w) = if own_is_white {
                            (&coords_opp, &coords_own)
                        } else {
                            (&coords_own, &coords_opp)
                        };
                        let eval = $eval_name.as_mut().unwrap();
                        eval.clear();
                        for (&b, &w) in coords_b.iter().zip(coords_w.iter()) {
                            eval.add(b).unwrap();
                            eval.add(w).unwrap();
                        }
                        if coords_b.len() > coords_w.len() {
                            for &b in &coords_b[coords_w.len()..] {
                                eval.add(b).unwrap();
                                eval.add(Coord::new()).unwrap(); // pass
                            }
                            eval.undo().unwrap();
                        } else if coords_w.len() > coords_b.len() {
                            for &w in &coords_w[coords_b.len()..] {
                                eval.add(Coord::new()).unwrap(); // pass
                                eval.add(w).unwrap();
                            }
                        }
                        if (eval.color_next().is_black() && own_is_white)
                            || (eval.color_next().is_white() && !own_is_white)
                        {
                            eval.add(Coord::new()).unwrap(); // pass
                        }
                        let timeout = info.timeout_turn((&eval).as_ref());
                        eval.set_turn_timeout(timeout);
                        let coord = eval.place_next();
                        if let Some((x, y)) = coord.get() {
                            println!("{x},{y}");
                        }
                    };
                }
                if is_size_20 {
                    process_board!(eval_20);
                } else {
                    process_board!(eval_15);
                }
            }
            "INFO" => {
                if let (Some(key), Some(val)) = (split_input.next(), split_input.next()) {
                    info.update(key, val);
                    if key == "rule" {
                        if !info.rule_supported() {
                            println!("ERROR - unsupported rule");
                            continue;
                        }
                        eval_switch_init!(); // this clears the record
                        // XXX: try to reload the previous record if exists
                    }
                }
                if let Some(eval_20) = eval_20.as_mut() {
                    eval_20.set_max_ram(info.ram_max());
                }
                if let Some(eval_15) = eval_15.as_mut() {
                    eval_15.set_max_ram(info.ram_max());
                }
            }
            "END" => {
                std::process::exit(0);
            }
            "ABOUT" => {
                println!(
                    "name=\"figrid-board\", version=\"0.3.0\", author=\"wuwbobo2021\", country=\"China\""
                );
            }
            _ => {
                println!("UNKNOWN");
                continue;
            }
        }
    }
}
