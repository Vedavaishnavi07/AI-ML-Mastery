import random


class Player:

    def __init__(self, name):
        self.name = name
        self.health = 100
        self.attack_power = 20
        self.inventory = []
        self.gold = 0
        self.has_key = False

    def show_status(self):
        print("\n----------------------------")
        print("Player:", self.name)
        print("Health:", self.health)
        print("Gold:", self.gold)
        print("Inventory:", self.inventory)
        print("Ancient Key:", self.has_key)
        print("----------------------------")

    def heal(self):

        if "Health Potion" in self.inventory:

            self.health += 30

            if self.health > 100:
                self.health = 100

            self.inventory.remove("Health Potion")

            print("You used a Health Potion!")

        else:

            print("No Health Potion available.")


class Enemy:

    def __init__(self, name, health, damage):

        self.name = name
        self.health = health
        self.damage = damage



def fight(player, enemy):

    print("\n", enemy.name, "appeared!")

    while enemy.health > 0 and player.health > 0:

        print("\nEnemy Health:", enemy.health)
        print("Your Health:", player.health)

        print("\n1. Attack")
        print("2. Use Potion")
        print("3. Special Sword Attack")
        print("4. Run")

        choice = input("Choose action: ").lower()


        if choice == "1" or choice == "attack":

            damage = random.randint(10, player.attack_power)

            enemy.health -= damage

            print("You attacked", enemy.name)
            print("Damage:", damage)


        elif choice == "2" or choice == "potion":

            player.heal()


        elif choice == "3" or choice == "sword":

            if "Sword" in player.inventory:

                damage = random.randint(30, 50)

                enemy.health -= damage

                print("Powerful Sword Strike!")
                print("Damage:", damage)

            else:

                print("You don't have a sword.")


        elif choice == "4" or choice == "run":

            print("You escaped!")

            return False


        else:

            print("Invalid choice")


        if enemy.health > 0:

            damage = random.randint(5, enemy.damage)

            player.health -= damage

            print(enemy.name, "attacked!")
            print("Damage taken:", damage)


    if player.health <= 0:

        print("\nYou were defeated.")

        return False


    print("\nYou defeated", enemy.name)

    reward = random.randint(50,150)

    player.gold += reward

    print("You received", reward, "gold")

    return True



def explore_forest(player):

    print("\nYou enter the forest.")

    event = random.choice(["potion","wolf","nothing"])


    if event == "potion":

        print("You found a Health Potion!")

        player.inventory.append("Health Potion")


    elif event == "wolf":

        fight(player, Enemy("Wolf",60,15))


    else:

        print("The forest is peaceful.")



def explore_cave(player):

    print("\nYou enter the ancient cave.")

    event = random.choice(["key","goblin","sword"])


    if event == "key":

        print("You found the Ancient Key!")

        player.has_key = True

        player.inventory.append("Ancient Key")


    elif event == "goblin":

        fight(player, Enemy("Goblin",80,20))


    else:

        print("You found a Sword!")

        player.inventory.append("Sword")

        player.attack_power += 10



def explore_river(player):

    print("\nYou reach the river.")

    event = random.choice(["snake","potion","bridge"])


    if event == "snake":

        fight(player, Enemy("Snake",50,10))


    elif event == "potion":

        print("You found a Health Potion!")

        player.inventory.append("Health Potion")


    else:

        print("You crossed the bridge safely.")



def final_temple(player):

    if not player.has_key:

        print("\nThe temple door is locked.")

        print("You need the Ancient Key.")

        return


    print("\nThe ancient temple opens.")

    print("A Dragon appears!")

    dragon = Enemy("Ancient Dragon",200,30)


    result = fight(player, dragon)


    if result:

        print("\n================================")
        print("LEGENDARY ENDING")
        print("You defeated the Dragon!")
        print("You became the Hero of the Lost Temple.")
        print("================================")

    else:

        print("\nThe Dragon defeated you.")

        print("Your adventure ends here.")



def main():

    print("="*45)
    print("WELCOME TO THE LOST TEMPLE ADVENTURE")
    print("="*45)


    name = input("Enter your name: ")

    player = Player(name)


    while True:

        player.show_status()


        if player.health <= 0:

            break


        print("\nChoose location")

        print("1. Forest")
        print("2. Cave")
        print("3. River")
        print("4. Final Temple")
        print("5. Exit")


        choice = input("\nEnter choice: ").lower()


        if choice == "1" or choice == "forest":

            explore_forest(player)


        elif choice == "2" or choice == "cave":

            explore_cave(player)


        elif choice == "3" or choice == "river":

            explore_river(player)


        elif choice == "4" or choice == "temple":

            final_temple(player)


        elif choice == "5" or choice == "exit":

            print("\nThanks for playing!")

            break


        else:

            print("Invalid choice")



if __name__ == "__main__":
    main()