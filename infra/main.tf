provider "aws" {
  region = "us-east-1"
}

resource "aws_ecr_repository" "repo" {
  name = "youtube-automation-repo"
}

resource "aws_ecs_cluster" "cluster" {
  name = "youtube-cluster"
}