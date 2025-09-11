import database.database_interaction as database_interaction

class Node:
    """A class representing a single node in the linked list."""
    def __init__(self, value, next_node=None):
        self.value = value 
        self.next_node = next_node

    def get_value(self):
        return self.value

    def get_next_node(self):
        return self.next_node

    def set_next_node(self, next_node):
        self.next_node = next_node


class LinkedList:
    """A class representing a generic linked list."""
    def __init__(self, value=None):
        self.head_node = Node(value) if value is not None else None

    def get_head_node(self):
        return self.head_node

    def insert_beginning(self, new_value):
        new_node = Node(new_value)
        new_node.set_next_node(self.head_node)
        self.head_node = new_node

    def insert_end(self, new_value):
        new_node = Node(new_value)
        if not self.head_node:
            self.head_node = new_node
            return
        current_node = self.head_node
        while current_node.get_next_node():
            current_node = current_node.get_next_node()
        current_node.set_next_node(new_node)

    def remove_node(self, value_to_remove):
        current_node = self.get_head_node()
        if not current_node:
            return
        if current_node.get_value() == value_to_remove:
            self.head_node = current_node.get_next_node()
        else:
            while current_node:
                next_node = current_node.get_next_node()
                if next_node and next_node.get_value() == value_to_remove:
                    current_node.set_next_node(next_node.get_next_node())
                    return
                current_node = next_node

    def stringify_list(self):
        """Returns a string representation of all values in the list."""
        string_list = ""
        current_node = self.get_head_node()
        while current_node:
            string_list += str(current_node.get_value()) + "\n"
            current_node = current_node.get_next_node()
        return string_list
    
    def export_multiple_pf_to_db(self, is_combined=False):
        """Exports all strategies in the linked list to the database."""
        current_node = self.get_head_node()
        while current_node:
            strat = current_node.get_value()
            print(f"Exporting strategy: {strat}")
            
            database_interaction.export_backtest_to_db(strat, is_combined=is_combined)

            current_node = current_node.get_next_node()

