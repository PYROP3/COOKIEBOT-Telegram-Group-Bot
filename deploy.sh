#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    print_status "Checking requirements..."
    
    if [ ! -f ".env" ]; then
        print_error ".env file not found. Please create it from .env.example"
        exit 1
    fi
    
    if [ ! -f "cookiebot-bucket-key.json" ]; then
        print_error "cookiebot-bucket-key.json not found. Please add your Google Cloud credentials"
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    print_success "All requirements satisfied"
}

build_image() {
    print_status "Building Docker image..."
    docker build -t cookiebot:latest .
    print_success "Docker image built successfully"
}

start_services() {
    print_status "Starting services..."
    
    docker compose up -d
    print_success "All services started successfully"
    
    print_status "Waiting for services to be healthy..."
    sleep 30
    
    # Check health status
    for i in {0..4}; do
        container_name="cookiebot-instance-$i"
        health_status=$(docker inspect --format='{{.State.Health.Status}}' $container_name 2>/dev/null || echo "unknown")
        if [ "$health_status" = "healthy" ]; then
            print_success "$container_name is healthy"
        else
            print_warning "$container_name health status: $health_status"
        fi
    done
}

stop_services() {
    print_status "Stopping all services..."
    docker compose down 2>/dev/null || true
    print_success "All services stopped"
}

show_status() {
    print_status "Service Status:"
    echo
    docker compose ps 2>/dev/null || echo "No services running"
    echo
    
    print_status "Resource Usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" $(docker ps --filter "name=cookiebot-instance" --format "{{.Names}}" 2>/dev/null) 2>/dev/null || echo "No containers running"
}

show_logs() {
    local service="$1"
    if [ -z "$service" ]; then
        print_status "Showing logs for all services..."
        docker compose logs -f --tail=50
    else
        print_status "Showing logs for $service..."
        docker compose logs -f --tail=50 "$service"
    fi
}

restart_service() {
    local service="$1"
    if [ -z "$service" ]; then
        print_error "Please specify a service name (cookiebot-0, cookiebot-1, etc.)"
        exit 1
    fi
    
    print_status "Restarting $service..."
    docker compose restart "$service"
    print_success "$service restarted successfully"
}

cleanup() {
    print_status "Cleaning up Docker resources..."
    docker system prune -f
    docker volume prune -f
    print_success "Cleanup completed"
}

case "$1" in
    "build")
        check_requirements
        build_image
        ;;
    "start")
        check_requirements
        build_image
        start_services
        ;;
    "stop")
        stop_services
        ;;
    "restart")
        stop_services
        check_requirements
        build_image
        start_services
        ;;
    "status")
        show_status
        ;;
    "logs")
        show_logs "$2"
        ;;
    "restart-service")
        restart_service "$2"
        ;;
    "cleanup")
        cleanup
        ;;
    "help"|"--help"|"-h")
        echo "COOKIEBOT Docker Deployment Script"
        echo
        echo "Usage: $0 [COMMAND] [OPTIONS]"
        echo
        echo "Commands:"
        echo "  build                 Build the Docker image"
        echo "  start                 Start all services"
        echo "  stop                  Stop all services"
        echo "  restart               Restart all services"
        echo "  status                Show service status and resource usage"
        echo "  logs [service]        Show logs (optionally for specific service)"
        echo "  restart-service NAME  Restart a specific service"
        echo "  scale NUMBER          Scale services (not implemented)"
        echo "  cleanup               Clean up Docker resources"
        echo "  help                  Show this help message"
        echo
        echo "Examples:"
        echo "  $0 start              # Start all services"
        echo "  $0 logs cookiebot-0   # Show logs for bot instance 0"
        echo "  $0 restart-service cookiebot-1  # Restart bot instance 1"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
