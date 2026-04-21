use std::{cmp::Ordering, default::Default, mem::MaybeUninit};

use super::Rec;
use crate::{Coord, Error};

const BOXES_CNT_MAX: usize = 5000;
const CUR_SEQ_MAX_CNT: usize = 26 * 26 + 16;

pub(crate) use nodes::Stack;
use nodes::{Node, NodePtr, NodePtrOpt, NodeStore};
mod nodes {
    use super::*;

    /// Bump storage used by a `Tree`.
    #[derive(Debug)]
    pub struct NodeStore<T> {
        boxes: [Option<Box<[MaybeUninit<T>]>>; BOXES_CNT_MAX],
        cnt_full_boxes: u16,
        cnt_units: u16,
    }

    /// Accessor of an item in the bump storage.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct NodePtr {
        lower: u16,
        upper: u16,
    }

    /// An option of `NodePtr`, similar to [std::option::Option].
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct NodePtrOpt {
        lower: u16,
        upper: u16,
    }

    impl NodePtrOpt {
        /// Returns a "none" value.
        #[inline(always)]
        pub const fn none() -> Self {
            Self {
                lower: u16::MAX,
                upper: u16::MAX,
            }
        }

        #[inline(always)]
        pub fn is_none(&self) -> bool {
            self.upper == u16::MAX && self.lower == u16::MAX
        }

        #[allow(unused)]
        #[inline(always)]
        pub fn is_some(&self) -> bool {
            !self.is_none()
        }

        /// Returns true if it equals `other`.
        #[inline(always)]
        pub fn eq_ptr(&self, other: NodePtr) -> bool {
            self.lower == other.lower && self.upper == other.upper
        }

        #[inline(always)]
        pub fn replace(&mut self, ptr: NodePtr) {
            self.upper = ptr.upper;
            self.lower = ptr.lower;
        }

        #[inline(always)]
        pub fn take(&mut self) -> Option<NodePtr> {
            if self.is_none() {
                None
            } else {
                let ptr = NodePtr {
                    upper: self.upper,
                    lower: self.lower,
                };
                *self = Self::none();
                Some(ptr)
            }
        }

        #[inline(always)]
        pub fn get(&self) -> Option<NodePtr> {
            if self.is_none() {
                None
            } else {
                Some(NodePtr {
                    upper: self.upper,
                    lower: self.lower,
                })
            }
        }

        #[inline(always)]
        pub fn set(&mut self, ptr: Option<NodePtr>) {
            if let Some(ptr) = ptr {
                self.upper = ptr.upper;
                self.lower = ptr.lower;
            } else {
                *self = Self::none();
            }
        }

        #[allow(unused)]
        #[inline(always)]
        pub fn unwrap(&self) -> NodePtr {
            self.get().unwrap()
        }
    }

    impl<T> NodeStore<T> {
        /// # Safety
        ///
        /// Any `NodePtr` returned by [NodeStore::store] should only be used with
        /// the `NodeStore` that has created it.
        pub unsafe fn new() -> Self {
            let mut boxes = [const { None }; BOXES_CNT_MAX];
            boxes[0].replace(Self::new_box());
            Self {
                boxes,
                cnt_full_boxes: 0,
                cnt_units: 0,
            }
        }

        #[inline]
        pub fn store(&mut self, val: T) -> NodePtr {
            unsafe {
                self.boxes
                    .get_unchecked_mut(self.cnt_full_boxes as usize)
                    .as_mut()
                    .unwrap_unchecked()
                    .get_unchecked_mut(self.cnt_units as usize)
            }
            .write(val);

            let ptr = NodePtr {
                lower: self.cnt_units,
                upper: self.cnt_full_boxes,
            };

            if self.cnt_units < u16::MAX {
                unsafe { self.cnt_units = self.cnt_units.unchecked_add(1) };
            } else {
                if self.cnt_full_boxes + 1 >= BOXES_CNT_MAX as u16 {
                    panic!("out of store limit")
                }
                self.cnt_units = 0;
                self.cnt_full_boxes += 1;
                self.boxes[self.cnt_full_boxes as usize].replace(Self::new_box());
            }

            ptr
        }

        #[inline(always)]
        pub fn get(&self, ptr: NodePtr) -> &T {
            unsafe {
                self.boxes
                    .get_unchecked(ptr.upper as usize)
                    .as_ref()
                    .unwrap_unchecked()
                    .get_unchecked(ptr.lower as usize)
                    .assume_init_ref()
            }
        }

        #[inline(always)]
        pub fn get_mut(&mut self, ptr: NodePtr) -> &mut T {
            unsafe {
                self.boxes
                    .get_unchecked_mut(ptr.upper as usize)
                    .as_mut()
                    .unwrap_unchecked()
                    .get_unchecked_mut(ptr.lower as usize)
                    .assume_init_mut()
            }
        }

        /// Note: this doesn't count for possible heap usage of `T`.
        #[inline]
        pub fn ram_used(&self) -> usize {
            size_of::<Self>() + (self.cnt_full_boxes + 1) as usize * size_of::<[Option<T>; 65536]>()
        }

        fn new_box() -> Box<[MaybeUninit<T>]> {
            let mut vec = Vec::new();
            for _ in 0..65536 {
                vec.push(MaybeUninit::uninit());
            }
            vec.into_boxed_slice()
        }
    }

    #[test]
    fn node_store_test() {
        let mut storer = unsafe { NodeStore::new() };
        let mut ptrs = Vec::new();
        for i in 0..70000 {
            ptrs.push(storer.store(i));
        }
        let mut i = 1;
        for ptr in ptrs.into_iter().skip(1) {
            assert_eq!(storer.get(ptr), &i);
            i += 1;
        }
    }

    /// A node in the tree structure.
    #[derive(Clone, Debug)]
    pub struct Node<const SZ: usize, T: Default + Clone> {
        pub coord: Coord<SZ>,
        pub info: T,
        pub down: NodePtrOpt,
        pub right: NodePtrOpt,
    }

    impl<const SZ: usize, T: Default + Clone> Node<SZ, T> {
        pub fn new() -> Self {
            assert!(SZ >= 5 && SZ <= 26);
            Self {
                coord: Coord::<SZ>::new(),
                info: T::default(),
                down: NodePtrOpt::none(),
                right: NodePtrOpt::none(),
            }
        }
    }

    /// An efficient stack structure without heap usage.
    pub struct Stack<T> {
        arr: [MaybeUninit<T>; CUR_SEQ_MAX_CNT],
        len: u16,
    }

    impl<T> Stack<T> {
        pub fn new() -> Self {
            Self {
                arr: unsafe { std::mem::zeroed() },
                len: 0u16,
            }
        }

        #[inline(always)]
        pub fn is_empty(&self) -> bool {
            self.len == 0
        }

        #[allow(unused)]
        #[inline(always)]
        pub fn top(&self) -> Option<&T> {
            if self.is_empty() {
                return None;
            }
            Some(unsafe { self.arr[self.len as usize - 1].assume_init_ref() })
        }

        #[allow(unused)]
        #[inline(always)]
        pub fn top_mut(&mut self) -> Option<&mut T> {
            if self.is_empty() {
                return None;
            }
            Some(unsafe { self.arr[self.len as usize - 1].assume_init_mut() })
        }

        #[allow(unused)]
        #[inline(always)]
        pub fn push(&mut self, item: T) {
            self.arr[self.len as usize].write(item);
            self.len += 1;
        }

        #[inline(always)]
        pub fn pop(&mut self) -> Option<T> {
            if self.is_empty() {
                return None;
            }
            self.len -= 1;
            Some(unsafe { self.arr[self.len as usize].assume_init_read() })
        }
    }

    #[test]
    fn stack_test() {
        let mut stack = Stack::new();
        assert_eq!(stack.pop(), None);
        stack.push(0);
        stack.push(1);
        assert_eq!(stack.pop(), Some(1));
        assert_eq!(stack.top(), Some(&0));
        *stack.top_mut().unwrap() = 2;
        assert_eq!(stack.pop(), Some(2));
        assert_eq!(stack.pop(), None);
        assert_eq!(stack.top(), None);
    }

    impl<T> Drop for Stack<T> {
        fn drop(&mut self) {
            if !std::mem::needs_drop::<T>() {
                return;
            }
            for item in &mut self.arr[..self.len as usize] {
                let _ = unsafe { item.assume_init_read() };
            }
        }
    }
}

/// Generic tree structure for the Five-in-a-Row game.
///
/// `SZ` is the board size; `T` is the carried information of each node.
///
/// This is a single-threaded model; to operate within the tree, the user can move
/// the internal cursor which is always located at a node at any timepoint. Each node
/// stores a [Coord] and a `T`, an optional left child and an optional right sibling.
///
/// Notes:
/// - The tree itself doesn't check for repeated coordinates in any possible route.
/// - [Tree::compress] can be called to clear deleted nodes in the internal bump storage.
pub struct Tree<const SZ: usize, T: Default + Clone> {
    // NOTE: don't pass `NodePtr` across different trees, otherwise there would be UBs!
    storage: NodeStore<Node<SZ, T>>,
    root: NodePtr,
    cur: NodePtr,
    cur_seq: [NodePtr; CUR_SEQ_MAX_CNT], // maximum index is `cur_dep`
    cur_dep: u16,
}

impl<const SZ: usize, T: Default + Clone> Tree<SZ, T> {
    /// Creates an empty tree and sets the cursor at the root node.
    pub fn new() -> Self {
        // SAFETY: Don't pass `NodePtr` across different trees!
        let mut storage = unsafe { NodeStore::new() };
        let root = storage.store(Node::new());
        Self {
            storage,
            root,
            cur: root,
            cur_seq: std::array::from_fn(|_| root),
            cur_dep: 0,
        }
    }

    /// Returns the RAM usage of the tree, including the internal bump storage.
    /// Note: this doesn't count for possible heap usage of `T`.
    pub fn ram_used(&self) -> usize {
        size_of::<Self>() + self.storage.ram_used()
    }

    #[inline(always)]
    fn access(&self, ptr: NodePtr) -> &Node<SZ, T> {
        self.storage.get(ptr)
    }

    #[inline(always)]
    fn access_mut(&mut self, ptr: NodePtr) -> &mut Node<SZ, T> {
        self.storage.get_mut(ptr)
    }

    #[inline(always)]
    fn new_node(&mut self) -> NodePtr {
        self.storage.store(Node::new())
    }

    #[inline(always)]
    fn cur_node(&self) -> &Node<SZ, T> {
        self.access(self.cur)
    }

    #[inline(always)]
    fn cur_node_mut(&mut self) -> &mut Node<SZ, T> {
        self.access_mut(self.cur)
    }

    /// Returns the depth of the current cursor. Depth of the root node is 0.
    #[inline(always)]
    pub fn cur_depth(&self) -> u16 {
        self.cur_dep
    }

    /// Returns the [Coord] of the cursored node.
    #[inline(always)]
    pub fn cur_coord(&self) -> Coord<SZ> {
        self.cur_node().coord
    }

    /// Returns a reference of the carried information of the cursored node.
    #[inline(always)]
    pub fn cur_info(&self) -> &T {
        &self.cur_node().info
    }

    /// Returns a mutable reference of the carried information of the cursored node.
    #[inline(always)]
    pub fn cur_info_mut(&mut self) -> &mut T {
        &mut self.cur_node_mut().info
    }

    /// Returns true if the cursored node is a leaf node.
    #[inline(always)]
    pub fn cur_is_leaf(&self) -> bool {
        self.cur_node().down.is_none()
    }

    /// Returns true if the cursored node is not a leaf node.
    #[inline(always)]
    pub fn cur_has_down(&self) -> bool {
        self.cur_node().down.is_some()
    }

    /// Returns true if the cursored node has at least one right node.
    #[inline(always)]
    pub fn cur_has_right(&self) -> bool {
        self.cur_node().right.is_some()
    }

    /// Returns true if the left child of the cursored node is a leaf node.
    #[inline(always)]
    pub fn down_is_leaf(&self) -> Option<bool> {
        let down = self.cur_node().down.get()?;
        Some(self.access(down).down.is_none())
    }

    /// Tries to get the information from the left child of the cursored node.
    #[inline(always)]
    pub fn down_info(&self) -> Option<&T> {
        let down = self.cur_node().down.get()?;
        Some(&self.access(down).info)
    }

    /// Returns the amount of child nodes of the cursored node.
    #[inline]
    pub fn cur_get_degree(&self) -> u16 {
        let Some(down) = self.cur_node().down.get() else {
            return 0;
        };
        let mut degree = 1;
        let mut cur = self.access(down);
        while let Some(right) = cur.right.get() {
            cur = self.access(right);
            degree += 1;
        }
        degree
    }

    /// Moves the current cursor to the root node.
    #[inline(always)]
    pub fn cur_goto_root(&mut self) {
        self.cur = self.root;
        self.cur_dep = 0;
    }

    /// Moves the current cursor to the parent node. Returns `Error::CursorAtEnd`
    /// if the current node is the root node.
    #[inline(always)]
    pub fn cur_go_up(&mut self) -> Result<(), Error> {
        if self.cur_dep == 0 {
            return Err(Error::CursorAtEnd);
        }
        self.cur_dep -= 1;
        self.cur = self.cur_seq[self.cur_dep as usize];
        Ok(())
    }

    /// Goes back through the current route until a node of `coord` is found. Does nothing
    /// and returns `Error::ItemNotExist` if `coord` does not exist in the route.
    #[inline(always)]
    pub fn cur_back_to(&mut self, coord: Coord<SZ>) -> Result<u16, Error> {
        for i in (0..=self.cur_depth()).rev() {
            if self.access(self.cur_seq[i as usize]).coord == coord {
                let d_depth = self.cur_depth() - i;
                while self.cur_depth() > i {
                    self.cur_go_up().unwrap();
                }
                return Ok(d_depth);
            }
        }
        Err(Error::ItemNotExist)
    }

    /// Moves the current cursor to the left child. Returns `Error::CursorAtEnd`
    /// if the current node is a leaf node.
    #[inline(always)]
    pub fn cur_go_down(&mut self) -> Result<(), Error> {
        let Some(down) = self.cur_node().down.get() else {
            return Err(Error::CursorAtEnd);
        };
        self.cur = down;
        self.cur_dep += 1;
        self.cur_seq[self.cur_dep as usize] = self.cur;
        Ok(())
    }

    /// Moves the current cursor to the right sibling. Returns `Error::CursorAtEnd`
    /// if the current node is the rightmost node.
    #[inline(always)]
    pub fn cur_go_right(&mut self) -> Result<(), Error> {
        let Some(right) = self.cur_node().right.get() else {
            return Err(Error::CursorAtEnd);
        };
        self.cur = right;
        self.cur_seq[self.cur_dep as usize] = self.cur;
        Ok(())
    }

    /// Finds a child node with `coord`.
    pub fn cur_find_next(&self, coord: Coord<SZ>) -> Option<&T> {
        let down = self.cur_node().down.get()?;
        let mut tmp_cur = self.access(down);
        while tmp_cur.coord != coord {
            let right = tmp_cur.right.get()?;
            tmp_cur = self.access(right);
        }
        Some(&tmp_cur.info)
    }

    /// Finds a child node by the comparer function in children of cursored node.
    pub fn cur_find_max_child_by(&self, f: impl Fn(&T, &T) -> Ordering) -> Option<(Coord<SZ>, &T)> {
        let mut cur_find = self.access(self.cur_node().down.get()?);
        let mut max = cur_find;
        while let Some(right) = cur_find.right.get() {
            cur_find = self.access(right);
            if f(&max.info, &cur_find.info) == Ordering::Less {
                max = cur_find;
            }
        }
        Some((max.coord, &max.info))
    }

    /// Finds a child node with the minimum value in children of cursored node.
    #[inline(always)]
    pub fn cur_find_min_child(&self) -> Option<(Coord<SZ>, &T)>
    where
        T: Ord,
    {
        self.cur_find_max_child_by(|a, b| b.cmp(a))
    }

    /// Finds a child node with the maximum value in children of cursored node.
    #[inline(always)]
    pub fn cur_find_max_child(&self) -> Option<(Coord<SZ>, &T)>
    where
        T: Ord,
    {
        self.cur_find_max_child_by(|a, b| a.cmp(b))
    }

    /// Sets the current cursor at a child node with `coord`, creating such a node
    /// if it did not exist.
    pub fn cur_set_next(&mut self, coord: Coord<SZ>) {
        if self.cur_go_down().is_err() {
            let node = self.new_node();
            self.cur_node_mut().down.replace(node);
            self.cur_go_down().unwrap();
        } else {
            while self.cur_coord() != coord {
                if self.cur_go_right().is_err() {
                    break;
                }
            }
            if self.cur_coord() == coord {
                return;
            }
            let node = self.new_node();
            self.cur_node_mut().right.replace(node);
            self.cur_go_right().unwrap();
        }
        self.cur_node_mut().coord = coord;
    }

    /// Makes the child with `coord` the left child of the cursored node if it exists.
    pub fn cur_adj_left_child(&mut self, coord: Coord<SZ>) {
        if self.cur_go_down().is_err() {
            return;
        }
        if self.cur_coord() == coord {
            self.cur_go_up().unwrap();
            return;
        }
        let mut p_left = self.cur;
        while self.cur_go_right().is_ok() {
            if self.cur_coord() == coord {
                let p_cur = self.cur;
                let p_right = self.cur_node_mut().right;
                self.access_mut(p_left).right = p_right;
                self.cur_go_up().unwrap();
                let prev_leftmost = self.cur_node().down;
                self.access_mut(p_cur).right = prev_leftmost;
                self.cur_node_mut().down.replace(p_cur);
                return;
            }
            p_left = self.cur;
        }
    }

    /// Efficient function for resetting children of the cursored node.
    ///
    /// Note: this doesn't check for repeated coordinates in `group`,
    /// which may lead to various unexpected troubles!
    pub fn cur_set_children<'a>(&mut self, group: impl IntoIterator<Item = &'a (Coord<SZ>, T)>)
    where
        T: 'a,
    {
        let mut group = group.into_iter();

        // delete all child nodes
        if self.cur_go_down().is_ok() {
            self.cur_go_up().unwrap();
            let _ = self.cur_node_mut().down.take();
        }

        // apply first item and goto this child node
        if let Some(&(coord, ref info)) = group.next() {
            self.cur_set_next(coord);
            self.cur_node_mut().info = info.clone();
        } else {
            return;
        }

        // add other items at right
        for &(coord, ref info) in group {
            let node = self.new_node();
            self.cur_node_mut().right.replace(node);
            self.cur_go_right().unwrap();
            let tmp_cur = self.cur_node_mut();
            tmp_cur.coord = coord;
            tmp_cur.info = info.clone();
        }

        self.cur_go_up().unwrap();
    }

    /// Deletes the cursored node and all of it descendents, then goes to the parent node.
    /// If it is a root node, it sets the root's information to a default value.
    pub fn cur_delete(&mut self) {
        let cur = self.cur;
        let right = self.cur_node_mut().right.take();

        if self.cur_go_up().is_err() {
            *self.cur_info_mut() = T::default();
            return;
        }
        if self.cur_node().down.eq_ptr(cur) {
            self.cur_node_mut().down.set(right);
            return;
        }
        self.cur_go_down().unwrap();
        while !self.cur_node().right.eq_ptr(cur) {
            self.cur_go_right().unwrap();
        }
        self.cur_node_mut().right.set(right);
        self.cur_go_up().unwrap();
    }

    /// Deletes left and right siblings of the cursored node.
    pub fn cur_delete_siblings(&mut self) {
        if self.cur_depth() == 0 {
            return;
        }
        self.cur_node_mut().right = NodePtrOpt::none();
        let cur = self.cur;
        self.cur_go_up().unwrap();
        self.cur_node_mut().down.replace(cur);
        self.cur_go_down().unwrap();
    }

    /// Establishes and walks through a route (starting from root) in the tree
    /// according to the `rec` forcefully.
    pub fn enter_rec(&mut self, rec: &impl Rec<SZ>) {
        self.cur_goto_root();
        for &coord in rec.iter() {
            self.cur_set_next(coord);
        }
    }

    /// Compresses the internal bump storage. Note: in the worst case, the peak RAM usage
    /// may be doubled during compression.
    pub fn compress(&mut self) {
        *self = self.clone();
    }

    /// Deletes everything and goes to the root node, as if it is newly created.
    pub fn clear(&mut self) {
        self.cur_goto_root();
        self.cur_delete();
        self.compress();
    }
}

impl<const SZ: usize, T: Default + Clone> Clone for Tree<SZ, T> {
    fn clone(&self) -> Self {
        let seq_coords: Vec<_> = self.cur_seq[1..=self.cur_dep as usize]
            .iter()
            .map(|&p| self.access(p).coord)
            .collect();

        let mut new_tree = Tree::new();
        *new_tree.cur_info_mut() = self.cur_info().clone();

        let mut stack = Stack::new();
        let mut cur = self.access(self.root);
        let mut cur_depth = 0_u16;
        loop {
            // Preorder: reach next node with `cur`, place new cursor at its parent's clone
            if let Some(down_old) = cur.down.get() {
                if let Some(right_old) = cur.right.get() {
                    stack.push((cur_depth, self.access(right_old)));
                }
                cur = self.access(down_old);
                cur_depth += 1;
            } else if let Some(right_old) = cur.right.get() {
                cur = self.access(right_old);
                new_tree.cur_go_up().unwrap();
            } else if let Some((depth, top_right_old)) = stack.pop() {
                cur = top_right_old;
                cur_depth = depth;
                while new_tree.cur_depth() > depth - 1 {
                    new_tree.cur_go_up().unwrap();
                }
            } else {
                break;
            }

            new_tree.cur_set_next(cur.coord);
            *new_tree.cur_info_mut() = cur.info.clone();
        }

        new_tree.cur_goto_root();
        for coord in seq_coords {
            new_tree.cur_set_next(coord);
        }
        new_tree
    }
}

impl<const SZ: usize, T: Default + Clone> Default for Tree<SZ, T> {
    fn default() -> Self {
        Self::new()
    }
}

#[test]
fn tree_del_node_test() {
    let mut tree = Tree::<15, _>::new();
    *tree.cur_info_mut() = 10;
    tree.cur_set_children(&[
        (Coord::from(7, 6), 10),
        (Coord::from(8, 5), 3),
        (Coord::from(8, 8), 1),
    ]);
    {
        let mut tree = tree.clone();
        tree.cur_go_down().unwrap();
        tree.cur_delete();
        assert_eq!(tree.cur_info(), &10);
        tree.cur_go_down().unwrap();
        assert_eq!(tree.cur_coord(), Coord::from(8, 5));
        tree.cur_go_up().unwrap();
        tree.cur_set_next(Coord::from(10, 9));
        tree.cur_go_up().unwrap();
        tree.cur_set_next(Coord::from(8, 5));
        assert_eq!(tree.cur_info(), &3);
        tree.cur_delete();
        tree.cur_go_down().unwrap();
        assert_eq!(tree.cur_coord(), Coord::from(8, 8));
        tree.cur_go_right().unwrap();
        assert_eq!(tree.cur_coord(), Coord::from(10, 9));
        tree.cur_delete();
        tree.cur_go_down().unwrap();
        tree.cur_delete();
        assert!(tree.cur_go_down().is_err());
    }
    {
        let mut tree = tree.clone();
        tree.cur_set_next(Coord::from(7, 6));
        tree.cur_delete_siblings();
        tree.cur_go_up().unwrap();
        assert!(tree.cur_find_next(Coord::from(8, 5)).is_none());
        assert!(tree.cur_find_next(Coord::from(8, 8)).is_none());
    }
    {
        let mut tree = tree.clone();
        tree.cur_set_next(Coord::from(8, 5));
        tree.cur_delete_siblings();
        tree.cur_go_up().unwrap();
        assert!(tree.cur_find_next(Coord::from(7, 6)).is_none());
        assert!(tree.cur_find_next(Coord::from(8, 8)).is_none());
    }
    {
        let mut tree = tree.clone();
        tree.cur_set_next(Coord::from(8, 8));
        tree.cur_delete_siblings();
        tree.cur_go_up().unwrap();
        assert!(tree.cur_find_next(Coord::from(7, 6)).is_none());
        assert!(tree.cur_find_next(Coord::from(8, 5)).is_none());
    }
}

#[test]
fn tree_adj_left_child_test() {
    let mut tree = Tree::<15, _>::new();
    tree.cur_set_children(&[
        (Coord::from(7, 6), 3),
        (Coord::from(8, 5), 5),
        (Coord::from(8, 8), 1),
    ]);

    let coord_max = tree.cur_find_max_child().unwrap();
    tree.cur_adj_left_child(coord_max.0);
    tree.cur_go_down().unwrap();
    assert_eq!(tree.cur_coord(), Coord::from(8, 5));

    tree.cur_go_right().unwrap();
    let coord = tree.cur_coord();
    tree.cur_go_up().unwrap();
    tree.cur_adj_left_child(coord);
    tree.cur_go_down().unwrap();
    assert_eq!(tree.cur_coord(), Coord::from(7, 6));
    tree.cur_go_right().unwrap();
    assert_eq!(tree.cur_coord(), Coord::from(8, 5));
    tree.cur_go_up().unwrap();

    tree.cur_adj_left_child(Coord::from(8, 8));
    tree.cur_go_down().unwrap();
    assert_eq!(tree.cur_coord(), Coord::from(8, 8));
    tree.cur_go_right().unwrap();
    assert_eq!(tree.cur_coord(), Coord::from(7, 6));
    tree.cur_go_right().unwrap();
    assert_eq!(tree.cur_coord(), Coord::from(8, 5));
}

#[test]
fn tree_compress_test() {
    let mut tree = Tree::<15, _>::new();
    *tree.cur_info_mut() = 16;
    tree.compress();

    assert_eq!(tree.cur_coord(), Coord::new());
    assert_eq!(tree.cur_info(), &16);
    assert!(tree.cur_go_down().is_err());
    assert!(tree.cur_go_right().is_err());

    tree.cur_set_next("j11".parse().unwrap());
    tree.cur_set_next("g7".parse().unwrap());
    tree.cur_set_children(&[
        (Coord::from(7, 6), 10),
        (Coord::from(8, 5), 0),
        (Coord::from(8, 8), 1),
    ]);
    tree.cur_go_up().unwrap();
    tree.cur_set_next("g9".parse().unwrap());
    tree.cur_set_children(&[(Coord::from(7, 7), 11), (Coord::from(6, 10), 3)]);
    tree.compress();

    assert_eq!(tree.cur_coord(), "g9".parse().unwrap());
    tree.cur_go_up().unwrap();
    assert_eq!(tree.cur_coord(), "j11".parse().unwrap());
    tree.cur_set_next("g9".parse().unwrap());
    tree.cur_go_down().unwrap();
    assert_eq!(tree.cur_coord(), Coord::from(7, 7));
    assert_eq!(tree.cur_info(), &11);
    tree.cur_go_right().unwrap();
    assert_eq!(tree.cur_coord(), Coord::from(6, 10));
    assert_eq!(tree.cur_info(), &3);
    tree.cur_goto_root();
    tree.cur_set_next("j11".parse().unwrap());
    assert!(tree.cur_find_next("g7".parse().unwrap()).is_some());
    assert!(tree.cur_find_next("g9".parse().unwrap()).is_some());

    tree.cur_set_next("g9".parse().unwrap());
    tree.cur_go_down().unwrap();
    tree.cur_back_to("j11".parse().unwrap()).unwrap();
}
